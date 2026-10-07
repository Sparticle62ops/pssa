#!/usr/bin/env python3
"""Offline, no-training audit of the real TUI in a controlling pseudo-terminal.

Run after building, e.g. through csrun:
  cargo build --release && python3 scripts/tui_audit_pty.py --dump /tmp/tui-screens.txt
No auth files are read or copied. Every session uses private temporary HOME,
config, chat and chain directories. Actions that could train are not sent.
"""
import argparse
import codecs
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import tempfile
import termios
import time
import unicodedata
from pathlib import Path

TABS = ["monitor", "chain", "model", "feed", "inference", "setup", "HF login",
        "Kaggle", "memory", "runs", "benchmark", "sample", "hardware", "math",
        "devices", "limits", "library", "mixer", "eval", "HF backup", "cloud log",
        "phone ping", "updates", "support", "GitHub", "sweeps", "timeline"]
SIZES = [(80, 24), (120, 40), (60, 20)]
ESC = b"\x1b"
UP, DOWN = b"\x1b[A", b"\x1b[B"


class Screen:
    """Small incremental VT screen reader for ratatui's cursor-addressed output."""
    def __init__(self, width, height):
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending = ""
        self.resize(width, height)

    def resize(self, width, height):
        self.width, self.height = width, height
        self.cells = [[" "] * width for _ in range(height)]
        self.x = self.y = 0

    def feed(self, data):
        self.pending += self.decoder.decode(data)
        s, i = self.pending, 0
        while i < len(s):
            c = s[i]
            if c == "\x1b":
                if i + 1 >= len(s):
                    break
                if s[i + 1] == "[":
                    j = i + 2
                    while j < len(s) and not ("@" <= s[j] <= "~"):
                        j += 1
                    if j == len(s):
                        break
                    body, final = s[i + 2:j], s[j]
                    nums = [int(v or "1") for v in body.lstrip("?").split(";")
                            if not v or v.isdigit()] or [1]
                    if final in "Hf":
                        self.y = min(self.height - 1, max(0, nums[0] - 1))
                        self.x = min(self.width - 1, max(0, (nums + [1])[1] - 1))
                    elif final == "J":
                        if nums[0] in (2, 3):
                            self.cells = [[" "] * self.width for _ in range(self.height)]
                        elif nums[0] == 0:
                            for y in range(self.y, self.height):
                                for x in range(self.x if y == self.y else 0, self.width):
                                    self.cells[y][x] = " "
                    elif final == "K":
                        for x in range(0 if nums[0] in (1, 2) else self.x,
                                       self.width if nums[0] in (0, 2) else self.x + 1):
                            self.cells[self.y][x] = " "
                    elif final in "ABCD":
                        if final == "A": self.y = max(0, self.y - nums[0])
                        if final == "B": self.y = min(self.height - 1, self.y + nums[0])
                        if final == "C": self.x = min(self.width - 1, self.x + nums[0])
                        if final == "D": self.x = max(0, self.x - nums[0])
                    i = j + 1
                    continue
                if s[i + 1] == "]":
                    j = i + 2
                    while j < len(s) and s[j] != "\x07" and s[j:j + 2] != "\x1b\\":
                        j += 1
                    if j == len(s):
                        break
                    i = j + (1 if s[j] == "\x07" else 2)
                    continue
                i += 2
                continue
            if c == "\r": self.x = 0
            elif c == "\n": self.y = min(self.height - 1, self.y + 1)
            elif c == "\b": self.x = max(0, self.x - 1)
            elif c == "\t": self.x = min(self.width - 1, (self.x // 8 + 1) * 8)
            elif c.isprintable():
                if unicodedata.combining(c):
                    if self.x: self.cells[self.y][self.x - 1] += c
                else:
                    wide = unicodedata.east_asian_width(c) in "WF"
                    self.cells[self.y][self.x] = c
                    if wide and self.x + 1 < self.width:
                        self.cells[self.y][self.x + 1] = ""
                    self.x = min(self.width - 1, self.x + 1 + int(wide))
            i += 1
        self.pending = s[i:]

    def text(self):
        return "\n".join("".join(row).rstrip() for row in self.cells)


class App:
    def __init__(self, binary, env, root, size, implicit=False):
        self.master, slave = pty.openpty()
        self.screen = Screen(*size)
        self.set_size(*size)
        args = [str(binary)] if implicit else [str(binary), "tui", "--chain", str(root / "chain"),
                                                "--chats-dir", str(root / "chats")]
        def controlling_terminal():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        self.child = subprocess.Popen(args, stdin=slave, stdout=slave, stderr=slave,
                                      env=env, preexec_fn=controlling_terminal)
        os.close(slave)
        self.drain(0.4)

    def set_size(self, width, height):
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        self.screen.resize(width, height)

    def drain(self, seconds=0.14):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ready, _, _ = select.select([self.master], [], [], max(0, min(0.03, end - time.monotonic())))
            if ready:
                try: self.screen.feed(os.read(self.master, 65536))
                except OSError: break

    def send(self, keys, seconds=0.14):
        assert self.child.poll() is None, f"TUI exited early: {self.child.returncode}"
        os.write(self.master, keys)
        self.drain(seconds)

    def open_tab(self, label):
        self.send(b"\x0b")
        self.send(("Open " + label).encode() + b"\r")

    def close(self, keys=b"\x03"):
        if self.child.poll() is None:
            self.send(keys, 0.2)
        try:
            result = self.child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait()
            raise AssertionError("TUI did not quit")
        finally:
            os.close(self.master)
        assert result == 0, f"TUI failed: exit {result}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "release/pssa")
    parser.add_argument("--dump", type=Path, default=Path("/tmp/pssa-tui-audit-screens.txt"))
    args = parser.parse_args()
    binary = args.binary.resolve()
    assert binary.is_file(), f"Build first: {binary}"
    dumps = []
    def capture(app, label):
        text = app.screen.text()
        assert "panicked" not in text, text
        dumps.append(f"--- {label} ({app.screen.width}x{app.screen.height}) ---\n{text}\n")
    with tempfile.TemporaryDirectory(prefix="pssa-tui-audit-") as scratch:
        root = Path(scratch)
        (root / "chain").mkdir()
        (root / "config").mkdir()
        env = os.environ.copy()
        for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
                    "KAGGLE_USERNAME", "KAGGLE_KEY", "PSSA_EVAL_PROMPTS", "OXIDE_EVAL_PROMPTS"]:
            env.pop(key, None)
        env.update(TERM="xterm-256color", HOME=str(root), XDG_CONFIG_HOME=str(root / "config"),
                   PSSA_OFFLINE="1", HF_HUB_OFFLINE="1", RAYON_NUM_THREADS="1")
        app = App(binary, env, root, SIZES[0])
        try:
            for size in SIZES:
                app.set_size(*size)
                app.drain(0.2)
                for label in TABS:
                    app.open_tab(label)
                    capture(app, label)
            # Modal overlays and all graph controls. No training is launched.
            app.open_tab("monitor")
            for key in [b"g", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"+", b"=", b"-", b"0",
                        b"\x1b[D", b"\x1b[C", b"\x1b[17~", b"\x1b[18~"]:
                app.send(key)
            app.send(b"?")
            for key in [DOWN, UP, b"\x1b[6~", b"\x1b[5~", b"\x1b[F", b"\x1b[H"]:
                app.send(key)
            capture(app, "help")
            app.send(ESC)
            app.send(b"\x0bno-such-command")
            capture(app, "empty palette")
            app.send(ESC)
            # Wizard pages, editing/cancellation, and equivalent command preview.
            app.open_tab("setup")
            for page in range(4):
                capture(app, f"setup page {page + 1}")
                if page < 3: app.send(b"\x1b[15~")
            app.send(b"c")
            app.send(b"\x1b[6~")
            capture(app, "setup command preview")
            app.send(b"c")
            app.send(b"\x1b[D" * 3 + DOWN + b"\r")
            app.send(b"draft? q \xe4\xb8\x96\xe7\x95\x8c")
            capture(app, "setup editor")
            app.send(ESC)
            for tab in ["limits", "library", "eval", "benchmark", "HF backup", "cloud log", "phone ping", "sweeps"]:
                app.open_tab(tab)
                app.send(b"p" if tab == "eval" else b"m" if tab == "library" else b"\r")
                capture(app, tab + " editor")
                app.send(ESC)
            # Unavailable device rows must not silently choose CPU or crash.
            app.open_tab("devices")
            app.send(b"\x1b[H" + DOWN + b"\r")
            assert "selected: cpu" in app.screen.text(), app.screen.text()
            app.send(DOWN + b"\r", 0.3)
            capture(app, "WebGPU unavailable")
            assert "Unavailable" in app.screen.text(), app.screen.text()
            assert "selected: cpu" in app.screen.text(), app.screen.text()
            app.send(DOWN + b"\r")
            capture(app, "CUDA unavailable")
            assert "Unavailable" in app.screen.text(), app.screen.text()
            assert "selected: cpu" in app.screen.text(), app.screen.text()
            # Function-key screen access, inference empty/error paths, tiny resize.
            for key in [b"\x1b[19~", b"\x1b[20~", b"\x1b[21~", b"\x1b[23~", b"\x1b[24~"]:
                app.send(key)
                capture(app, "function key " + repr(key))
            app.open_tab("inference")
            for command in ["/help", "/model", "/chats", "/speech", "hello"]:
                app.send(b"\x15" + command.encode() + b"\r")
                capture(app, "inference " + command)
            app.send(ESC)
            app.set_size(20, 8)
            app.drain(0.2)
            capture(app, "tiny inference")
        finally:
            app.close()
        # Dedicated quit paths: browse Escape, editor Escape only cancels, q quits,
        # and implicit CLI launch uses the TUI instead of the non-TTY home page.
        for tab in ["monitor", "setup", "devices", "library", "HF backup", "timeline"]:
            app = App(binary, env, root, SIZES[0])
            app.open_tab(tab)
            app.close(ESC)
        app = App(binary, env, root, SIZES[0], implicit=True)
        capture(app, "implicit CLI launch")
        app.close(b"q")
        assert not (root / "chain" / "train.log").exists()
    args.dump.write_text("\n".join(dumps))
    print(f"PASS: {len(dumps)} captured screens, all 27 tabs at {SIZES}; dump: {args.dump}")
    for dump in dumps:
        if any(marker in dump.splitlines()[0] for marker in ["unavailable", "help", "setup editor", "tiny inference"]):
            print(dump)


if __name__ == "__main__":
    main()
