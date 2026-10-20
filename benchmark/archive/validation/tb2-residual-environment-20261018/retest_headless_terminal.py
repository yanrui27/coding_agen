import pexpect
import os
import re

from base_terminal import BaseTerminal


class HeadlessTerminal(BaseTerminal):
    def __init__(self, env=None):
        """
        Initialize a headless terminal with an interactive bash shell.

        Args:
            env: Optional environment dict to pass to the shell.
        """
        if env is None:
            env = os.environ.copy()
        # Use /bin/bash -i to start an interactive shell, which sources ~/.bashrc
        self._child = pexpect.spawn(
            '/bin/bash',
            ['-i'],
            encoding='utf-8',
            timeout=30,
            env=env,
            echo=False,
        )
        # Wait for the shell prompt to appear to confirm bash is ready
        self._child.expect(
            [r'\$\s*$', r'#\s*$', r'\[\d*m\]', r'\$'],
            timeout=10,
        )

    def send_keystrokes(self, keystrokes: str, wait_sec: float = 0.0) -> None:
        """
        Sends keystrokes to the terminal.

        Args:
            keystrokes: The keystrokes to send to the terminal.
            wait_sec: The number of seconds to wait for the command to complete.
        """
        # Process keystrokes character by character to properly handle
        # control characters and special keys
        i = 0
        while i < len(keystrokes):
            char = keystrokes[i]
            code = ord(char)

            if code == 0:
                # Null character
                self._child.sendcontrol('@')
                i += 1
            elif code == 9:  # Tab
                self._child.sendcontrol('I')
                i += 1
            elif code == 127:  # Backspace
                self._child.sendcontrol('H')
                i += 1
            elif code == 27:  # Escape
                self._child.sendcontrol('[')
                i += 1
            elif code == 13:  # Carriage return
                self._child.sendcontrol('M')
                i += 1
            elif code < 32:  # Other control characters (e.g., Ctrl+C = \x03)
                self._child.sendcontrol(chr(code + 64))
                i += 1
            elif char == '\n':  # Newline / Enter
                self._child.sendline('')
                i += 1
            else:
                self._child.send(char)
                i += 1

        # If wait_sec is specified, wait for the shell to return to prompt
        if wait_sec > 0:
            try:
                self._child.expect(
                    [r'\$\s*$', r'#\s*$', r'\[\d*m\]', r'\$'],
                    timeout=wait_sec,
                )
            except pexpect.TIMEOUT:
                pass