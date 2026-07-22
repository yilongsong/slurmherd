"""Getting work onto a cluster.

A transport answers one question: *run this batch of agent operations over
there and give me the results*. Two implementations:

``LocalTransport``
    Calls the agent in-process. Used when you are already on the machine, and
    by the test suite.
``SSHTransport``
    Ships the agent over ``ssh`` and reads the JSON back.

Both expose exactly the same surface, so no code above this layer knows or
cares whether a cluster is local or three time zones away.

On SSH multiplexing
-------------------
University clusters almost universally require two-factor auth, and paying a
Duo push per ``squeue`` would make the tool unusable. slurmherd therefore uses
OpenSSH connection sharing: ``slurmherd connect <cluster>`` opens one master
connection interactively -- you approve 2FA once -- and every later operation
rides that socket in milliseconds, for as long as ``control_persist`` says.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import agent
from .errors import SlurmherdError
from .models import Connection
from .util import ensure_dir


class TransportError(SlurmherdError):
    """A cluster could not be reached, or answered with something unusable."""


class AuthRequired(TransportError):
    """The cluster needs an interactive login before it can be driven."""


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------


class Transport:
    """Runs :mod:`slurmherd.agent` operations somewhere."""

    #: Human-readable location, used in error messages.
    location = "local"

    def batch(self, ops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    # -- conveniences ----------------------------------------------------

    def one(self, op: Dict[str, Any]) -> Dict[str, Any]:
        results = self.batch([op])
        return results[0] if results else {"error": "no result"}

    def ping(self) -> Dict[str, Any]:
        return self.one({"op": "ping"})

    def run(
        self, cmd: str, cwd: Optional[str] = None, timeout: int = 120
    ) -> Dict[str, Any]:
        return self.one({"op": "run", "cmd": cmd, "cwd": cwd, "timeout": timeout})

    def read(self, path: str, tail: Optional[int] = None) -> Dict[str, Any]:
        return self.one({"op": "read", "path": path, "tail": tail})

    def write(self, path: str, text: str, mode: Optional[int] = None) -> Dict[str, Any]:
        return self.one({"op": "write", "path": path, "text": text, "mode": mode})

    def exists(self, path: str) -> bool:
        return bool(self.one({"op": "exists", "path": path}).get("exists"))

    def mkdir(self, path: str) -> Dict[str, Any]:
        return self.one({"op": "mkdir", "path": path})

    def glob(self, pattern: str) -> List[str]:
        return list(self.one({"op": "glob", "pattern": pattern}).get("paths", []))

    # -- lifecycle -------------------------------------------------------

    def is_connected(self) -> bool:
        return True

    def connect(self, interactive: bool = True) -> bool:
        return True

    def disconnect(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def push(self, source: str, dest: str, exclude: Sequence[str] = (), delete: bool = False):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Local
# --------------------------------------------------------------------------


class LocalTransport(Transport):
    """Runs agent operations in this process."""

    location = "local"

    def batch(self, ops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        response = agent.execute({"ops": list(ops)})
        return response.get("results", [])

    def push(self, source: str, dest: str, exclude: Sequence[str] = (), delete: bool = False):
        cmd = ["rsync", "-a", "--info=stats1"]
        for pattern in exclude:
            cmd += ["--exclude", pattern]
        if delete:
            cmd.append("--delete")
        cmd += [os.path.join(os.path.expanduser(source), ""), os.path.expanduser(dest)]
        return subprocess.run(cmd, capture_output=True, text=True)


# --------------------------------------------------------------------------
# SSH
# --------------------------------------------------------------------------

_BOOTSTRAP = (
    "import sys,base64,zlib;"
    "exec(zlib.decompress(base64.b64decode(sys.argv[1])).decode('utf-8'))"
)

_AUTH_HINTS = (
    "permission denied",
    "publickey",
    "password",
    "authentication failed",
    "no such identity",
    "host key verification failed",
    "duo",
)


class SSHTransport(Transport):
    """Runs agent operations on a remote host over a shared SSH connection."""

    def __init__(self, conn: Connection, control_dir: Optional[Path] = None) -> None:
        self.conn = conn
        self.location = conn.host if not conn.user else f"{conn.user}@{conn.host}"
        self._control_dir = Path(control_dir or Path.home() / ".slurmherd" / "ssh")
        self._payload: Optional[str] = None

    # -- ssh command construction ----------------------------------------

    @property
    def control_path(self) -> Optional[str]:
        if not self.conn.control_persist:
            return None
        ensure_dir(self._control_dir, mode=0o700)
        # %C is a hash of (host, port, user) -- short enough to stay under the
        # ~104 byte limit on unix socket paths, which %h/%r/%p routinely blow.
        return str(self._control_dir / "%C")

    def _ssh_base(self, interactive: bool = False) -> List[str]:
        cmd = ["ssh"]
        if self.conn.port:
            cmd += ["-p", str(self.conn.port)]
        if self.conn.identity_file:
            cmd += ["-i", os.path.expanduser(self.conn.identity_file)]
        if self.conn.proxy_jump:
            cmd += ["-J", self.conn.proxy_jump]
        control = self.control_path
        if control:
            cmd += [
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={control}",
                "-o", f"ControlPersist={self.conn.control_persist}",
            ]
        cmd += ["-o", f"ConnectTimeout={self.conn.connect_timeout}"]
        if not interactive:
            # Fail fast instead of blocking on a password or 2FA prompt that
            # nobody is watching; `slurmherd connect` is the interactive path.
            cmd += ["-o", "BatchMode=yes"]
        for option in self.conn.options:
            cmd += ["-o", option]
        target = f"{self.conn.user}@{self.conn.host}" if self.conn.user else self.conn.host
        cmd.append(target)
        return cmd

    def _agent_payload(self) -> str:
        if self._payload is None:
            raw = agent.source().encode("utf-8")
            self._payload = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
        return self._payload

    def _remote_command(self) -> str:
        return "{python} -c {boot} {payload}".format(
            python=self.conn.python,
            boot=shlex.quote(_BOOTSTRAP),
            payload=self._agent_payload(),
        )

    # -- connection management -------------------------------------------

    def is_connected(self) -> bool:
        """True when a multiplexed master connection is already open."""
        if not self.control_path:
            return False
        cmd = self._ssh_base() + ["-O", "check"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0

    def connect(self, interactive: bool = True) -> bool:
        """Open the shared master connection, prompting for 2FA if needed."""
        if self.is_connected():
            return True
        if not self.control_path:
            return True  # sharing disabled; every op authenticates on its own
        cmd = self._ssh_base(interactive=interactive)
        cmd = cmd[:-1] + ["-o", "ControlMaster=yes", "-N", "-f", cmd[-1]]
        try:
            proc = subprocess.run(cmd, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TransportError(f"could not reach {self.location}: {exc}") from exc
        return proc.returncode == 0

    def disconnect(self) -> bool:
        if not self.control_path:
            return True
        cmd = self._ssh_base() + ["-O", "exit"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0

    # -- the actual work -------------------------------------------------

    def batch(self, ops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        request = json.dumps({"ops": list(ops)})
        cmd = self._ssh_base() + [self._remote_command()]
        # Generous: the batch may contain a user probe command of its own, and
        # each op already enforces its own timeout on the far side.
        timeout = 60 + sum(int(op.get("timeout") or 0) for op in ops)
        try:
            proc = subprocess.run(
                cmd, input=request, capture_output=True, text=True, timeout=timeout
            )
        except FileNotFoundError as exc:
            raise TransportError("ssh not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise TransportError(
                f"{self.location}: no response after {timeout}s"
            ) from exc

        payload = _extract(proc.stdout)
        if payload is None:
            raise _diagnose(self.location, proc)
        if "fatal" in payload:
            raise TransportError(f"{self.location}: agent error: {payload['fatal']}")
        return payload.get("results", [])

    def push(self, source: str, dest: str, exclude: Sequence[str] = (), delete: bool = False):
        """rsync a local directory to the cluster over the shared connection."""
        if not shutil.which("rsync"):
            raise TransportError("rsync not found on PATH -- needed by `slurmherd push`")
        ssh_cmd = " ".join(shlex.quote(part) for part in self._ssh_base()[:-1])
        cmd = ["rsync", "-az", "--info=stats1", "-e", ssh_cmd]
        for pattern in exclude:
            cmd += ["--exclude", pattern]
        if delete:
            cmd.append("--delete")
        target = f"{self.conn.user}@{self.conn.host}" if self.conn.user else self.conn.host
        cmd += [os.path.join(os.path.expanduser(source), ""), f"{target}:{dest}"]
        return subprocess.run(cmd, capture_output=True, text=True)


def _extract(stdout: str) -> Optional[Dict[str, Any]]:
    """Pull the framed JSON response out of whatever the login node printed.

    Clusters print banners, quota warnings and MOTDs on every login; the
    sentinels are how we ignore all of it.
    """
    start = stdout.rfind(agent.BEGIN)
    if start < 0:
        return None
    end = stdout.find(agent.END, start)
    if end < 0:
        return None
    blob = stdout[start + len(agent.BEGIN) : end].strip()
    try:
        return json.loads(blob)
    except ValueError:
        return None


def _diagnose(location: str, proc: "subprocess.CompletedProcess[str]") -> TransportError:
    """Turn a failed SSH invocation into an error a human can act on."""
    stderr = (proc.stderr or "").strip()
    lowered = stderr.lower()
    tail = "\n".join(stderr.splitlines()[-6:])

    if any(hint in lowered for hint in _AUTH_HINTS):
        return AuthRequired(
            f"{location}: SSH authentication failed.\n{tail}\n\n"
            f"Run `slurmherd connect {location}` to log in once interactively "
            "(including any 2FA prompt); slurmherd reuses that connection afterwards."
        )
    if "command not found" in lowered or "No such file" in stderr:
        return TransportError(
            f"{location}: could not start the remote Python interpreter.\n{tail}\n\n"
            "Set `connect.python` for this cluster to a working interpreter "
            "(for example `/usr/bin/python3` or a module-provided one)."
        )
    stdout_tail = "\n".join((proc.stdout or "").strip().splitlines()[-6:])
    return TransportError(
        f"{location}: unexpected response from the cluster (exit {proc.returncode}).\n"
        f"--- stderr ---\n{tail or '(empty)'}\n"
        f"--- stdout ---\n{stdout_tail or '(empty)'}"
    )


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def make_transport(conn: Connection, control_dir: Optional[Path] = None) -> Transport:
    """Build the right transport for a connection spec."""
    if not conn.host or conn.host in ("local", "localhost", "127.0.0.1"):
        return LocalTransport()
    return SSHTransport(conn, control_dir=control_dir)
