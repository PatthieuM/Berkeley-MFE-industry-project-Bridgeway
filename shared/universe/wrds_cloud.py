"""Open a local PostgreSQL connection through an authenticated WRDS Cloud tunnel.

The helper reads the user's standard ``~/.pgpass`` entry, authenticates SSH
without printing the password, and forwards a temporary local port to the WRDS
PostgreSQL service. If WRDS requests MFA, it selects Duo Push; the user must
approve that notification. No credential is stored in this repository.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import socket
from typing import Iterator, Optional

import pexpect
import psycopg2
from psycopg2.extensions import connection as PostgreSQLConnection


DB_HOST = "wrds-pgdata.wharton.upenn.edu"
DB_PORT = 9737
DB_NAME = "wrds"
SSH_HOST = "wrds-cloud.wharton.upenn.edu"
TUNNEL_READY = "WRDS_TUNNEL_READY"


def _split_pgpass_line(line: str) -> list[str]:
    """Split a pgpass line while honoring escaped colons and backslashes."""
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line.rstrip("\n"):
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":" and len(fields) < 4:
            fields.append("".join(current))
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    fields.append("".join(current))
    return fields


def _load_credentials(pgpass_path: Path, username: Optional[str] = None) -> tuple[str, str]:
    """Return ``(username, password)`` for WRDS from a ``.pgpass`` file.

    Only lines for the WRDS PostgreSQL host, port and database are considered.
    With several WRDS users stored, ``username`` or ``WRDS_USERNAME`` must pick one.
    """
    if not pgpass_path.is_file():
        raise RuntimeError(f"WRDS credential file not found: {pgpass_path}")

    requested_username = username or os.environ.get("WRDS_USERNAME")
    matches: list[tuple[str, str]] = []
    for raw_line in pgpass_path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.lstrip().startswith("#"):
            continue
        fields = _split_pgpass_line(raw_line)
        if len(fields) != 5:
            continue
        host, port, database, entry_username, password = fields
        if (host, port, database) != (DB_HOST, str(DB_PORT), DB_NAME):
            continue
        if requested_username and entry_username != requested_username:
            continue
        matches.append((entry_username, password))

    unique_usernames = {entry_username for entry_username, _ in matches}
    if not matches:
        suffix = f" for username {requested_username}" if requested_username else ""
        raise RuntimeError(f"No matching WRDS entry in {pgpass_path}{suffix}")
    if len(unique_usernames) > 1:
        raise RuntimeError("Multiple WRDS users are stored; pass --username or set WRDS_USERNAME")
    return matches[-1]


def _available_local_port() -> int:
    """Ask the OS for a free local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@contextmanager
def _wrds_cloud_tunnel(
    username: str,
    password: str,
    *,
    ssh_host: str = SSH_HOST,
    authentication_timeout: int = 55,
) -> Iterator[int]:
    """Open an SSH tunnel to WRDS and yield the local port forwarding to PostgreSQL.

    Answers the SSH prompts (host-key confirmation, password, Duo option 1 =
    push) and yields once the remote side prints the ready marker. The SSH
    process is closed when the context exits.
    """
    local_port = _available_local_port()
    remote_command = f"printf {TUNNEL_READY}; sleep 3600"
    child = pexpect.spawn(
        "ssh",
        [
            "-o", "ConnectTimeout=20",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "PreferredAuthentications=keyboard-interactive,password",
            "-o", "PubkeyAuthentication=no",
            "-L", f"127.0.0.1:{local_port}:{DB_HOST}:{DB_PORT}",
            f"{username}@{ssh_host}",
            remote_command,
        ],
        encoding="utf-8",
        timeout=authentication_timeout,
    )
    patterns = [
        r"(?i)are you sure you want to continue connecting",
        r"(?i)password:",
        r"(?i)passcode or option.*:",
        TUNNEL_READY,
        r"(?i)permission denied",
        pexpect.EOF,
        pexpect.TIMEOUT,
    ]
    try:
        for _ in range(12):
            matched = child.expect(patterns)
            if matched == 0:
                child.sendline("yes")
            elif matched == 1:
                child.sendline(password)
            elif matched == 2:
                child.sendline("1")
            elif matched == 3:
                yield local_port
                return
            elif matched == 4:
                raise RuntimeError("WRDS SSH authentication was denied")
            elif matched == 5:
                raise RuntimeError("WRDS SSH connection closed before the tunnel was ready")
            else:
                raise RuntimeError("WRDS SSH authentication timed out; approve Duo and retry")
        raise RuntimeError("Unexpected WRDS SSH authentication loop")
    finally:
        child.close(force=True)


@contextmanager
def open_wrds_connection(
    username: Optional[str] = None,
    *,
    pgpass_path: Optional[Path] = None,
) -> Iterator[PostgreSQLConnection]:
    """Yield a psycopg2 connection routed securely through WRDS Cloud."""
    credentials_path = pgpass_path or (Path.home() / ".pgpass")
    resolved_username, password = _load_credentials(credentials_path, username)
    with _wrds_cloud_tunnel(resolved_username, password) as local_port:
        connection = psycopg2.connect(
            host="127.0.0.1",
            port=local_port,
            dbname=DB_NAME,
            user=resolved_username,
            password=password,
            sslmode="require",
            connect_timeout=30,
            application_name="Bridgeway WRDS Cloud tunnel",
        )
        try:
            yield connection
        finally:
            connection.close()


def main() -> int:
    """Open a WRDS tunnel and print a minimal connection health check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", help="WRDS username; inferred from ~/.pgpass when unique")
    args = parser.parse_args()
    with open_wrds_connection(args.username) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_user, current_database()")
            username, database = cursor.fetchone()
    print(f"WRDS connection OK: user={username}, database={database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
