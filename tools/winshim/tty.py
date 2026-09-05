"""Windows shim for stdlib `tty`, which imports termios and so fails on Windows."""
from termios import TCSAFLUSH, error  # noqa: F401


def setraw(fd, when=TCSAFLUSH):
    raise error("tty.setraw unavailable on Windows (interactive TTY path only)")


def setcbreak(fd, when=TCSAFLUSH):
    raise error("tty.setcbreak unavailable on Windows (interactive TTY path only)")
