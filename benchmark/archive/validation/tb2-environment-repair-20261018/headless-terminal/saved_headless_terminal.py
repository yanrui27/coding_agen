import pexpect
import time
import os
import sys

from base_terminal import BaseTerminal


class HeadlessTerminal(BaseTerminal):
    """
    A headless terminal that mimics an interactive bash shell.

    Starts as an interactive bash shell which sources ~/.bashrc.
    Supports sending keystrokes including modifier/control keys.
    """

    PROMPT = r'\$'

    def __init__(self, bash: str = '/bin/bash', **kwargs):
        """
        Initialize the HeadlessTerminal by starting an interactive bash shell.

        Args:
            bash: Path to the bash executable.
            **kwargs: Additional keyword arguments passed to pexpect.spawn.
        """
        self._child = pexpect.spawn(
            bash,
            ['--rcfile', '/dev/null'],
            encoding='utf-8',
            timeout=30,
            **kwargs,
        )
        # Give bash a moment to start
        self._child.expect([r'BASH_VERSION|PS1|\$|#', pexpect.TIMEOUT], timeout=10)

        # Source ~/.bashrc for an interactive bash session
        self.send_keystrokes('source ~/.bashrc\n', wait_sec=2.0)
        # Clear any trailing prompt characters
        self.send_keystrokes('\n', wait_sec=0.5)

    @property
    def child(self):
        """Access the underlying pexpect child process."""
        return self._child

    def send_keystrokes(self, keystrokes: str, wait_sec: float = 0.0) -> None:
        """
        Sends keystrokes to the terminal.

        Args:
            keystrokes: The keystrokes to send to the terminal.
            wait_sec: The number of seconds to wait for the command to complete.
        """
        if not keystrokes:
            return

        self._child.send(keystrokes)

        if wait_sec > 0:
            time.sleep(wait_sec)

    def sendline(self, line: str = '', wait_sec: float = 0.0) -> None:
        """
        Sends a line of text followed by a newline to the terminal.

        Args:
            line: The text to send.
            wait_sec: The number of seconds to wait for the command to complete.
        """
        if line:
            self._child.sendline(line)
        else:
            self._child.sendline()

        if wait_sec > 0:
            time.sleep(wait_sec)

    def read(self, timeout: float = 1.0) -> str:
        """
        Read all available output from the terminal.

        Args:
            timeout: How long to wait for output.

        Returns:
            The output string read from the terminal.
        """
        try:
            return self._child.read_nonblocking(4096, timeout=timeout)
        except pexpect.TIMEOUT:
            return ''

    def read_until(self, pattern, timeout: float = 1.0) -> str:
        """
        Read output until a pattern is matched.

        Args:
            pattern: The pattern to match (string or compiled regex).
            timeout: How long to wait for the pattern.

        Returns:
            The output string read from the terminal.
        """
        try:
            self._child.expect(pattern, timeout=timeout)
            if hasattr(self._child, 'match') and self._child.match:
                return self._child.before + self._child.match.group(0)
            return self._child.before
        except pexpect.TIMEOUT:
            return self._child.before

    def isalive(self) -> bool:
        """Check if the terminal process is still alive."""
        try:
            return self._child.isalive()
        except Exception:
            return False

    def close(self, force: bool = False) -> None:
        """
        Close the terminal.

        Args:
            force: If True, forcefully terminate the process.
        """
        if self.isalive():
            if force:
                self._child.kill(9)
            else:
                self.send_keystrokes('\x04')  # Ctrl+D
            self._child.wait()

    @property
    def pid(self) -> int:
        """Get the PID of the terminal process."""
        return self._child.pid

    @property
    def exitstatus(self):
        """Get the exit status of the terminal process."""
        return self._child.exitstatus

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass