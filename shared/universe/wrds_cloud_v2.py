"""Opt-in WRDS connector with verified host keys and a context-lived tunnel.

The published-study helper remains in wrds_cloud.py. Enroll the server's verified
host key before using this version. No connection is made when importing it.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import time

import pexpect
import psycopg2

if __package__:
    from .wrds_cloud import DB_HOST, DB_PORT, DB_NAME, SSH_HOST, TUNNEL_READY, _split_pgpass_line, _available_local_port
else:
    from wrds_cloud import DB_HOST, DB_PORT, DB_NAME, SSH_HOST, TUNNEL_READY, _split_pgpass_line, _available_local_port


def _load_credentials(path: Path, username: str | None = None) -> tuple[str, str]:
    """Select the first matching pgpass entry, supporting wildcard fields."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise RuntimeError(f"WRDS credential file not found: {path}")
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise RuntimeError("pgpass permissions must exclude group and other access (chmod 600)")
    requested = username or os.environ.get("WRDS_USERNAME")
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = _split_pgpass_line(line)
        if len(fields) != 5:
            continue
        if all(entry in ("*", actual) for entry, actual in zip(fields[:3], (DB_HOST, str(DB_PORT), DB_NAME))):
            entries.append(fields)
    if not requested:
        users = {entry[3] for entry in entries if entry[3] != "*"}
        if len(users) != 1:
            raise RuntimeError("Pass a WRDS username or set WRDS_USERNAME")
        requested = users.pop()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", requested):
        raise ValueError("invalid WRDS username")
    for entry in entries:
        if entry[3] in (requested, "*"):
            return requested, entry[4]
    raise RuntimeError("No matching WRDS pgpass entry")


@contextmanager
def _wrds_cloud_tunnel(username, password, *, ssh_host=SSH_HOST, known_hosts_path=None,
                       duo_option="1", authentication_timeout=55):
    """Authenticate with a known host and hold SSH forwarding until context exit."""
    if authentication_timeout <= 0 or not re.fullmatch(r"[0-9]+", str(duo_option)):
        raise ValueError("use a positive authentication timeout and a numeric Duo option")
    local_port = _available_local_port()
    options = ["-T", "-o", "StrictHostKeyChecking=yes", "-o", "ExitOnForwardFailure=yes",
               "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]
    if known_hosts_path is not None:
        path = Path(known_hosts_path).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError("known_hosts file missing; enroll a verified server host key first")
        options += ["-o", f"UserKnownHostsFile={path}"]
    # The authenticated remote command reports readiness, then waits for stdin
    # to close. It imposes no fixed one-hour lifetime.
    options += ["-L", f"127.0.0.1:{local_port}:{DB_HOST}:{DB_PORT}", f"{username}@{ssh_host}",
                f"printf {TUNNEL_READY}; cat >/dev/null"]
    child = pexpect.spawn("ssh", options, encoding="utf-8", timeout=authentication_timeout)
    patterns = [r"(?i)password:", r"(?i)passcode or option.*:", TUNNEL_READY,
                r"(?i)host key verification failed|remote host identification has changed|no .* host key is known",
                r"(?i)permission denied", pexpect.EOF, pexpect.TIMEOUT]
    deadline = time.monotonic() + authentication_timeout
    prompts = 0
    try:
        while child.isalive() and time.monotonic() < deadline:
            matched = child.expect(patterns, timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            if matched in (0, 1):
                prompts += 1
                if prompts > 12:
                    raise RuntimeError("Too many WRDS authentication prompts")
                child.sendline(password if matched == 0 else str(duo_option))
            elif matched == 2:
                yield local_port
                return
            elif matched == 3:
                raise RuntimeError("WRDS host key is not verified; check the fingerprint before enrollment")
            elif matched == 4:
                raise RuntimeError("WRDS SSH authentication was denied")
            elif matched == 5:
                raise RuntimeError("WRDS SSH closed before forwarding was ready")
        raise RuntimeError("WRDS authentication did not complete; check host-key setup and Duo approval")
    finally:
        child.close(force=True)


@contextmanager
def open_wrds_connection(username=None, *, pgpass_path=None, known_hosts_path=None, duo_option="1"):
    credentials = Path(pgpass_path or os.environ.get("PGPASSFILE", Path.home() / ".pgpass"))
    user, password = _load_credentials(credentials, username)
    with _wrds_cloud_tunnel(user, password, known_hosts_path=known_hosts_path, duo_option=duo_option) as port:
        connection = psycopg2.connect(host="127.0.0.1", port=port, dbname=DB_NAME,
                                      user=user, password=password, sslmode="require",
                                      connect_timeout=30, application_name="Bridgeway WRDS connector v2")
        try:
            yield connection
        finally:
            connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username")
    parser.add_argument("--pgpass", type=Path)
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--duo-option", default="1")
    args = parser.parse_args(argv)
    with open_wrds_connection(args.username, pgpass_path=args.pgpass,
                              known_hosts_path=args.known_hosts, duo_option=args.duo_option) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            if cursor.fetchone() != (1,):
                raise RuntimeError("WRDS health check failed")
    print("WRDS connection OK (connector v2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
