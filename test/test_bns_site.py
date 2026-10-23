import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bns_site import BnsSite, UPLOAD_CONFIG


class FakeSite:
    """The BNS endpoints a recording upload uses, plus the storage URL the site hands out."""

    def __init__(self, jobs=None):
        self.jobs = jobs or []
        self.stored = {}
        self.calls = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body, status=200):
                data = json.dumps(body).encode()
                self.send_response(status)
                if self.path == "/api/admin/login":
                    self.send_header("Set-Cookie", "bns_session=abc; Path=/")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def body(self):
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))

            def do_GET(self):
                site.calls.append(("GET", self.path))
                if self.path == "/api/jobs":
                    return self.reply(site.jobs)
                if self.path.startswith("/api/database/rest/v1/transcriptions"):
                    return self.reply([{"title": "October 21st, 2026", "summary": "A day in London."}])
                self.reply({"error": "unknown"}, 404)

            def do_PUT(self):
                site.calls.append(("PUT", self.path))
                site.stored[self.path] = (self.body(), self.headers.get("Content-Type"))
                self.reply({})

            def do_POST(self):
                payload = json.loads(self.body() or b"null")
                site.calls.append(("POST", self.path, payload))
                if self.path == "/api/admin/login":
                    return self.reply({"isAdmin": True})
                if self.headers.get("Cookie") != "bns_session=abc":
                    return self.reply({"error": "Admin login required."}, 403)
                if self.path == "/api/audio/upload-url":
                    return self.reply({"key": "audio/uploads/k/a.mp3", "uploadUrl": site.base + "/storage/a.mp3",
                                       "headers": {"Content-Type": "audio/mpeg", "Cache-Control": "public"}})
                if self.path == "/api/jobs":
                    job = {"id": payload["id"], "fileName": payload["fileName"], "status": "completed",
                           "progress": 100, "message": "Archived", "recordId": payload["id"]}
                    site.jobs.insert(0, job)
                    return self.reply(job, 202)
                if self.path.endswith("/retry"):
                    site.jobs[0].update(status="completed", message="Archived")
                    return self.reply(site.jobs[0])
                self.reply({"error": "unknown"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


class UploadRecordingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.file = Path(self.folder.name) / "0000420_2026-10-21.mp3"
        self.file.write_bytes(b"ID3 audio")

    def tearDown(self):
        self.folder.cleanup()

    def test_new_recording_goes_to_storage_then_the_job_queue(self):
        fake = FakeSite()
        try:
            site = BnsSite(fake.base)
            site.login("admin", "pw")
            result = site.upload_recording(self.file, 60, log=lambda *_: None)
        finally:
            fake.server.shutdown()
        self.assertEqual(result["status"], "uploaded")
        self.assertEqual(result["title"], "October 21st, 2026")
        self.assertEqual(fake.stored["/storage/a.mp3"], (b"ID3 audio", "audio/mpeg"))
        queued = next(call[2] for call in fake.calls if call[:2] == ("POST", "/api/jobs"))
        self.assertEqual(queued["audioKey"], "audio/uploads/k/a.mp3")
        self.assertEqual(queued["fileSize"], len(b"ID3 audio"))
        self.assertEqual(queued["config"], UPLOAD_CONFIG)
        self.assertNotIn("locations", queued, "the site fills in the known places itself")

    def test_failed_job_from_an_earlier_run_is_retried_without_uploading_again(self):
        fake = FakeSite(jobs=[{"id": "11111111-1111-4111-8111-111111111111", "fileName": self.file.name,
                               "status": "error", "progress": 40, "message": "Processing failed"}])
        try:
            site = BnsSite(fake.base)
            site.login("admin", "pw")
            result = site.upload_recording(self.file, 60, log=lambda *_: None)
        finally:
            fake.server.shutdown()
        self.assertEqual(result["status"], "uploaded")
        self.assertFalse(fake.stored, "no second upload")
        self.assertIn(("POST", "/api/jobs/11111111-1111-4111-8111-111111111111/retry", None), fake.calls)


if __name__ == "__main__":
    unittest.main()
