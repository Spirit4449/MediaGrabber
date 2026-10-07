"""Load the saved Telegram login without letting workers write to its database."""

import contextlib
import sqlite3
from pathlib import Path

from telethon.crypto import AuthKey
from telethon.sessions import MemorySession


def load_download_session(session_path):
    path = Path(session_path)
    if not str(path).endswith(".session"):
        path = Path(str(path) + ".session")
    # Read-only access also prevents creation of an empty database if login is missing.
    with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro")) as db:
        row = db.execute(
            "SELECT dc_id, server_address, port, auth_key FROM sessions LIMIT 1"
        ).fetchone()
    if not row or not row[3]:
        raise ValueError("Saved Telegram session has no login; run downloader.py --login first")
    session = MemorySession()
    session.set_dc(*row[:3])
    session.auth_key = AuthKey(row[3])
    return session
