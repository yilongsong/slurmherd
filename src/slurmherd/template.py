"""``{{ name }}`` substitution.

Deliberately not a template language. There are no loops, no conditionals and
no expressions -- only names looked up in a namespace. That keeps job scripts
predictable and makes every unresolved name a config error you see at
``validate`` time rather than a mystery at 3am on a compute node.

``{{ ... }}`` is used rather than ``${...}`` so that shell variables in your
commands (``$HOME``, ``${SLURM_JOB_ID}``) pass through untouched.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional, Set

from .errors import ConfigError, did_you_mean

TOKEN_RE = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")

MAX_PASSES = 10


def _lookup(name: str, ns: Mapping[str, Any]) -> Any:
    if name in ns:
        return ns[name]
    # Support dotted access into nested mappings, e.g. {{ site.scratch }}.
    head, _, rest = name.partition(".")
    if rest and isinstance(ns.get(head), Mapping):
        cursor: Any = ns[head]
        for part in rest.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                raise KeyError(name)
            cursor = cursor[part]
        return cursor
    raise KeyError(name)


def render(
    text: str,
    ns: Mapping[str, Any],
    *,
    defer: Iterable[str] = (),
    path: Optional[str] = None,
    key: str = "",
) -> str:
    """Substitute ``{{ name }}`` tokens in ``text``.

    Names in ``defer`` are left in place for a later pass -- used for values
    like ``attempt`` that are only known when a job script is written.

    A ``{{ ... }}`` whose contents are not a valid name (JSON braces in a
    command line, say) is left exactly as written.
    """
    deferred: Set[str] = set(defer)

    def replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        if not NAME_RE.match(name):
            return match.group(0)
        if name in deferred or name.split(".")[0] in deferred:
            return match.group(0)
        try:
            value = _lookup(name, ns)
        except KeyError:
            available = sorted(k for k in ns if not k.startswith("_"))
            guesses = did_you_mean(name, available)
            hint = (
                f"did you mean {{{{ {guesses[0]} }}}}?"
                if guesses
                else "available: " + ", ".join(available[:24]) + ("…" if len(available) > 24 else "")
            )
            raise ConfigError(f"undefined template name {{{{ {name} }}}}", path, key, hint) from None
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    result = text
    for _ in range(MAX_PASSES):
        rendered = TOKEN_RE.sub(replace, result)
        if rendered == result:
            return rendered
        result = rendered
    raise ConfigError(
        "template expansion did not settle -- a variable probably refers to itself",
        path,
        key,
    )


def render_deep(
    obj: Any,
    ns: Mapping[str, Any],
    *,
    defer: Iterable[str] = (),
    path: Optional[str] = None,
    key: str = "",
) -> Any:
    """Apply :func:`render` to every string leaf of a nested structure."""
    if isinstance(obj, str):
        return render(obj, ns, defer=defer, path=path, key=key)
    if isinstance(obj, Mapping):
        return {
            k: render_deep(v, ns, defer=defer, path=path, key=f"{key}.{k}" if key else str(k))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            render_deep(v, ns, defer=defer, path=path, key=f"{key}[{i}]")
            for i, v in enumerate(obj)
        ]
    return obj


def resolve_vars(
    raw: Mapping[str, Any], base: Mapping[str, Any], *, path: Optional[str] = None
) -> Dict[str, Any]:
    """Resolve a ``vars`` block whose entries may refer to each other.

    Iterates until stable so declaration order in the file does not matter.
    """
    resolved: Dict[str, Any] = dict(raw)
    for _ in range(MAX_PASSES):
        ns = {**base, **resolved}
        updated = {
            k: render_deep(v, ns, path=path, key=f"vars.{k}") for k, v in resolved.items()
        }
        if updated == resolved:
            return updated
        resolved = updated
    raise ConfigError("vars refer to each other in a cycle", path, "vars")


def find_tokens(text: str) -> Set[str]:
    """Return the set of valid names referenced by ``text``."""
    return {m for m in TOKEN_RE.findall(text) if NAME_RE.match(m)}
