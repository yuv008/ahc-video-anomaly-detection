"""Minimal Windows shim for the Unix-only `termios` module.

colab_cli imports termios at module load, but only calls it inside an interactive
attached-console path guarded by `is_tty` - which is False when driven non-interactively.
So satisfying the import is enough; these functions should never actually run.
"""
TCSANOW = 0
TCSADRAIN = 1
TCSAFLUSH = 2


class error(Exception):
    pass


def tcgetattr(fd):
    raise error("termios.tcgetattr unavailable on Windows (interactive TTY path only)")


def tcsetattr(fd, when, attributes):
    raise error("termios.tcsetattr unavailable on Windows (interactive TTY path only)")
