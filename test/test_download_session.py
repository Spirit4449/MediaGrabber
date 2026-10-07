import multiprocessing
import sqlite3
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from telethon.crypto import AuthKey
from telethon.sessions import SQLiteSession

from download_session import load_download_session


def run_worker(path):
    for _ in range(10):
        session = load_download_session(path)
        original = (session.dc_id, session.server_address, session.port, session.auth_key.key)
        session.set_dc(4, "149.154.167.91", 443)
        session.auth_key = AuthKey(b"b" * 256)
        session.save()
        session.close()
    return original


class DownloadSessionTests(unittest.TestCase):
    def test_concurrent_workers_with_active_database_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "login")
            saved = SQLiteSession(path)
            saved.set_dc(2, "149.154.167.51", 443)
            saved.auth_key = AuthKey(b"a" * 256)
            saved.save()
            # Telethon writers hold this lock until their next session.save().
            saved.set_dc(3, "149.154.175.100", 443)
            try:
                with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
                    results = list(pool.map(run_worker, [path] * 8))
                self.assertEqual(results, [(2, "149.154.167.51", 443, b"a" * 256)] * 8)
                saved._conn.rollback()
                loaded = load_download_session(path + ".session")
                self.assertEqual(loaded.dc_id, 2)
                self.assertEqual(loaded.auth_key.key, b"a" * 256)
            finally:
                saved.close()

    def test_missing_session_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.session"
            with self.assertRaises(sqlite3.OperationalError):
                load_download_session(path)
            self.assertFalse(path.exists())

    def test_session_without_login_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "empty")
            saved = SQLiteSession(path)
            saved.close()
            with self.assertRaisesRegex(ValueError, "--login"):
                load_download_session(path)


if __name__ == "__main__":
    unittest.main()
