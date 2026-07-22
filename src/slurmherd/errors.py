"""Error types.

Every error a user can trigger by writing a bad config should be a
``ConfigError`` carrying the file and key that caused it, so the CLI can print
something actionable instead of a traceback.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence


class SlurmherdError(Exception):
    """Base class for all errors raised by slurmherd."""


class ConfigError(SlurmherdError):
    """A configuration file is invalid.

    Args:
        message: What is wrong, phrased so the user can fix it.
        path: Config file the problem came from.
        key: Dotted key path within that file, e.g. ``experiments[2].command``.
        hint: Optional follow-up sentence suggesting the fix.
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        key: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.key = key
        self.hint = hint

    def __str__(self) -> str:
        where = ""
        if self.path and self.key:
            where = f"{self.path}: {self.key}: "
        elif self.path:
            where = f"{self.path}: "
        elif self.key:
            where = f"{self.key}: "
        text = f"{where}{self.message}"
        if self.hint:
            text += f"\n  hint: {self.hint}"
        return text


class SchedulerError(SlurmherdError):
    """A scheduler command failed or returned something unparseable."""


class StateError(SlurmherdError):
    """The on-disk state store is unusable (corrupt, locked, unwritable)."""


class UsageError(SlurmherdError):
    """The user asked for something that does not make sense."""


def did_you_mean(word: str, options: Iterable[str], limit: int = 3) -> Sequence[str]:
    """Return the closest matches to ``word`` from ``options``.

    Used to turn "unknown key 'commnad'" into a suggestion. Pure stdlib.
    """
    import difflib

    return difflib.get_close_matches(word, list(options), n=limit, cutoff=0.6)


def unknown_key_error(
    key: str, allowed: Iterable[str], path: Optional[str] = None, parent: str = ""
) -> ConfigError:
    """Build a ConfigError for an unrecognised config key, with suggestions."""
    allowed = sorted(allowed)
    guesses = did_you_mean(key, allowed)
    hint = f"did you mean {guesses[0]!r}?" if guesses else "known keys: " + ", ".join(allowed)
    full = f"{parent}.{key}" if parent else key
    return ConfigError(f"unknown key {key!r}", path=path, key=full, hint=hint)
