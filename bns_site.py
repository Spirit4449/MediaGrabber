"""
bns_site.py

Signed-in calls to the Brahma naa Sange site, for daily_bns_sync.py (recordings) and
moments_sync.py (Moments posts). Plain HTTP with a session cookie, the same endpoints the
site's own pages use; no browser.

A recording goes: POST /api/audio/upload-url -> PUT the file to the returned presigned R2 URL
-> POST /api/jobs (the site verifies the upload and queues processing) -> poll GET /api/jobs.
"""
from __future__ import annotations

import http.cookiejar
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# Cloudflare in front of the site rejects requests without a browser-like User-Agent.
USER_AGENT = "Mozilla/5.0 (compatible; MediaGrabber/1.0)"

# The site's default AI settings for uploads (DEFAULT_AI_CONFIG in BNS hooks/useAppController.tsx).
# Generation runs on AGY first; these models apply only when it falls back to the Gemini API.
# No keyType: the server picks a free key itself.
UPLOAD_CONFIG = {
    "transcriptionModel": "gemini-2.5-flash",
    "refinementModel": "gemini-2.5-flash",
    "analysisModel": "gemini-2.5-flash",
    "zeroResultModel": "gemini-2.5-flash-lite",
    "rerankModel": "gemini-2.5-flash",
    "rerankEnabled": True,
    "embeddingModel": "gemini-embedding-001",
    "meaningModel": "gemini-2.5-flash",
    "dynamicModelFallback": True,
    "dynamicApiFallback": True,
    "sophisticatedUpload": True,
}


def _truncate(value: str, max_len: int = 1400) -> str:
    value = (value or "").strip()
    return value if len(value) <= max_len else value[: max_len - 3].rstrip() + "..."


def job_status_line(job: dict) -> str:
    line = f"{job.get('status', '?')} {int(job.get('progress') or 0)}% - {job.get('message', '')}".strip()
    if job.get("error"):
        line = f"{line}\n{job['error']}"
    return line


class BnsSite:
    """Signed-in JSON calls to the BNS site with an admin login (ADMIN_CREDENTIALS on the site)."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def _call(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error") or detail
            except (ValueError, AttributeError):
                pass
            raise RuntimeError(f"BNS site {method} {path} failed (HTTP {exc.code}): {_truncate(str(detail), 300)}") from exc

    def login(self, username: str, password: str) -> None:
        result = self._call("POST", "/api/admin/login", {"username": username, "password": password})
        if not (result or {}).get("isAdmin"):
            raise RuntimeError(f"BNS account {username} is not an admin; it must be listed in the site's ADMIN_CREDENTIALS")

    # ---- Moments ----

    def import_posts(self, channel_id: int, username: str, posts: list[dict], complete: bool) -> dict:
        return self._call("POST", "/api/moments/import",
                          {"channelId": channel_id, "channelUsername": username, "posts": posts, "complete": complete})

    def status(self, message_ids: list[int]) -> list[dict]:
        return self._call("GET", "/api/moments/status?ids=" + ",".join(str(i) for i in message_ids)) or []

    # ---- Recordings ----

    def jobs_for(self, file_name: str) -> list[dict]:
        """This account's processing jobs for file_name, newest first."""
        jobs = self._call("GET", "/api/jobs") or []
        return [job for job in jobs if isinstance(job, dict) and job.get("fileName") == file_name]

    def put_audio(self, file_path: Path) -> str:
        """Uploads the file straight to storage and returns its key."""
        size = file_path.stat().st_size
        prepared = self._call("POST", "/api/audio/upload-url", {"fileName": file_path.name, "size": size})
        headers = {**(prepared.get("headers") or {}), "Content-Length": str(size)}
        with file_path.open("rb") as handle:
            request = urllib.request.Request(prepared["uploadUrl"], data=handle, method="PUT", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=max(self.timeout, 1800)) as response:
                    response.read()
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"Audio storage refused the upload (HTTP {exc.code}): {_truncate(exc.read().decode('utf-8', 'replace'), 300)}") from exc
        return prepared["key"]

    def queue_recording(self, file_path: Path, key: str) -> dict:
        return self._call("POST", "/api/jobs", {"id": str(uuid.uuid4()), "fileName": file_path.name,
                                                "fileSize": file_path.stat().st_size, "audioKey": key, "config": UPLOAD_CONFIG})

    def retry_job(self, job_id: str) -> dict:
        return self._call("POST", f"/api/jobs/{job_id}/retry")

    def record_summary(self, record_id: str) -> tuple[str, str]:
        query = urllib.parse.urlencode({"id": f"eq.{record_id}", "select": "title,summary:analysis->>summary"})
        try:
            rows = self._call("GET", f"/api/database/rest/v1/transcriptions?{query}") or []
        except RuntimeError:
            return "", ""
        if not rows:
            return "", ""
        return str(rows[0].get("title") or "").strip(), str(rows[0].get("summary") or "").strip()

    def upload_recording(self, file_path: Path, timeout_seconds: int, log=print) -> dict:
        """
        Uploads one recording and waits for the site to finish processing it.
        -> {"status": "uploaded"|"error", "reason", "title", "summary"}

        Processing runs on the site's server, so a job from an earlier attempt may still be running
        (or finished, or failed) after this script timed out or restarted; that job is reused, and a
        failed one retried, instead of uploading again.
        """
        def result(status: str, reason: str, title: str = "", summary: str = "") -> dict:
            return {"status": status, "reason": reason, "title": title, "summary": summary}

        file_name = file_path.name
        jobs = self.jobs_for(file_name)
        job = jobs[0] if jobs else None
        if job and job.get("status") in {"error", "stopped"}:
            log(f"Retrying existing BNS website job for {file_name}: {job_status_line(job)}")
            self.retry_job(job["id"])
        elif job:
            log(f"Found existing BNS website job for {file_name}: {job_status_line(job)}")
        else:
            key = self.put_audio(file_path)
            job = self.queue_recording(file_path, key)
            log(f"Queued {file_name} on the BNS website: {job_status_line(job)}")

        deadline = time.time() + max(60, timeout_seconds)
        next_log_at = time.time() + 60
        while time.time() < deadline:
            jobs = self.jobs_for(file_name)
            job = jobs[0] if jobs else job
            if job.get("status") == "completed":
                title, summary = self.record_summary(job.get("recordId") or job.get("id") or "")
                return result("uploaded", job.get("message") or "Completed", title, summary)
            if job.get("status") in {"error", "stopped"}:
                return result("error", _truncate(job_status_line(job), 1800) or "Website processing failed.")
            if time.time() >= next_log_at:
                log(f"BNS website still processing {file_name}: {_truncate(job_status_line(job), 500)}")
                next_log_at = time.time() + 60
            time.sleep(10)
        return result("error", f"Timed out after {timeout_seconds}s waiting for the site to finish {file_name}. "
                               "The job keeps running there; the next run picks it up without uploading again.\n\n"
                               f"Last status:\n{_truncate(job_status_line(job), 1800)}")
