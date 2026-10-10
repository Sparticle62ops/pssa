#!/usr/bin/env python3
"""Resolve the smoke manifest's binary without shell expansion."""
import os
import sys

binary = os.environ.get("PSSA_BIN", "target/fast/pssa")
os.execv(binary, [binary, *sys.argv[1:]])
