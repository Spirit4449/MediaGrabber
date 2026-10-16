#!/usr/bin/env python3
"""
daily_bns_sync.py

Fetch new media posts from a source channel and upload them to a target channel.

Default source: 1773081661
Default target: 3130614830

Behaviour:
- Maintains a small state file at .state/bns_state.json with last_id.
- On first run (no state), seeds the last_id to the latest message and exits to avoid backfilling.
  Use --seed to force processing of the recent window.

Requires TELEGRAM_API_ID, TELEGRAM_API_HASH and TELEGRAM_SESSION env variables (session file path).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from download_session import load_download_session
from reliability import load_state, save_state, exclusive_lock
from pathlib import Path

from telethon import TelegramClient, types, utils as tutils  # type: ignore

# Load environment variables from .env file (for automated runs)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Minimal MIME -> ext fallback
MIME_EXT_FALLBACK = {
    "video/mp4": ".mp4",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/x-m4a": ".m4a",
}

DEFAULT_SOURCE = 1773081661
DEFAULT_TARGET = 3130614830
STATE_DIR = Path(".state")
DEFAULT_STATE_FILE = STATE_DIR / "bns_state.json"
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")
BNS_MAX_UPLOAD_MB = float(os.environ.get("BNS_MAX_UPLOAD_MB", "49"))
BNS_MAX_UPLOAD_BYTES = int(BNS_MAX_UPLOAD_MB * 1024 * 1024)
AUDIO_EXTS = {".mp3", ".m4a", ".ogg", ".oga", ".wav", ".flac", ".aac", ".opus", ".m4b"}
DEFAULT_BNS_ADMIN_USERNAME = "admin"
DEFAULT_BNS_ADMIN_PASSWORD = "admin123"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clean_env_value(value: str) -> str:
    value = (value or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = _clean_env_value(value)
    return env


def _parse_admin_credentials(raw: str) -> list[tuple[str, str]]:
    credentials: list[tuple[str, str]] = []
    for pair in _clean_env_value(raw).split(","):
        if ":" not in pair:
            continue
        username, password = pair.split(":", 1)
        username = username.strip()
        password = password.strip()
        if username and password:
            credentials.append((username, password))
    return credentials


def _load_bns_site_admin_credentials() -> tuple[str, str] | None:
    env_path = os.environ.get("BNS_SITE_ENV_FILE", "").strip()
    candidates = [Path(env_path)] if env_path else []
    candidates.extend(
        [
            Path(__file__).resolve().parent.parent / "Brahma-naa-sange" / ".env.local",
            Path.home() / "Desktop" / "Brahma-naa-sange" / ".env.local",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)

        values = _read_env_file(resolved)
        for key in ("ADMIN_CREDENTIALS", "VITE_ADMIN_CREDENTIALS", "VITE_ADMIN_USERS"):
            parsed = _parse_admin_credentials(values.get(key, ""))
            if parsed:
                return parsed[0]

        username = values.get("VITE_ADMIN_USERNAME", "").strip()
        password = values.get("VITE_ADMIN_PASSWORD", "").strip()
        if username and password:
            return username, password
    return None


def _resolve_bns_admin_credentials() -> tuple[str, str]:
    username = os.environ.get("BNS_ADMIN_USERNAME", "").strip()
    password = os.environ.get("BNS_ADMIN_PASSWORD", "").strip()
    has_custom_env_credentials = bool(username and password) and (
        username != DEFAULT_BNS_ADMIN_USERNAME or password != DEFAULT_BNS_ADMIN_PASSWORD
    )
    if has_custom_env_credentials:
        return username, password

    site_credentials = _load_bns_site_admin_credentials()
    if site_credentials:
        return site_credentials

    return username, password


def _xpath_literal(value: str) -> str:
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    parts = value.split("'")
    return "concat(" + ", \"'\", ".join([f"'{p}'" for p in parts]) + ")"


def _truncate(value: str, max_len: int = 1400) -> str:
    value = (value or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def _tail_file(path: Path, max_lines: int = 40, max_len: int = 1800) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()[-max_lines:]
        return _truncate("".join(lines).strip(), max_len)
    except Exception:
        return ""


def _write_debug_artifacts(driver, label: str) -> str:
    debug_dir = os.environ.get("BNS_SELENIUM_DEBUG_DIR", "logs/selenium").strip()
    if not debug_dir:
        return ""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    base = f"{safe_label}-{stamp}"
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    html_path = debug_path / f"{base}.html"
    png_path = debug_path / f"{base}.png"
    with contextlib.suppress(Exception):
        html_path.write_text(driver.page_source, encoding="utf-8", errors="replace")
    with contextlib.suppress(Exception):
        driver.save_screenshot(str(png_path))
    return f"HTML: {html_path}, Screenshot: {png_path}"


def send_telegram_bot_message(bot_token: str, chat_id: str, text: str) -> None:
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": _truncate(text, 3500),
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=20) as res:
        if res.status >= 400:
            raise RuntimeError(f"Telegram Bot API HTTP {res.status}")


def notify(bot_token: str, chat_id: str, text: str) -> None:
    text = _truncate(text, 3500)
    print(text)
    if not bot_token or not chat_id:
        return
    try:
        send_telegram_bot_message(bot_token, chat_id, text)
    except Exception as exc:
        print(f"Telegram notify failed: {exc}")


def _browser_error_excerpt(driver, max_len: int = 900) -> str:
    try:
        entries = driver.get_log("browser")
    except Exception:
        return ""

    interesting = []
    for entry in entries[-20:]:
        level = str(entry.get("level", "")).upper()
        message = str(entry.get("message", "")).strip()
        lowered = message.lower()
        if (
            level in {"SEVERE", "ERROR"}
            or "429" in lowered
            or "quota" in lowered
            or "resource_exhausted" in lowered
        ):
            interesting.append(message)

    if not interesting:
        return ""
    return _truncate("\n".join(interesting[-3:]), max_len)


# Runs in the signed-in page, so the request carries the site's session cookie.
_SITE_FETCH_SCRIPT = """
const [path, method, done] = arguments;
fetch(path, {
  method,
  credentials: 'same-origin',
  ...(method === 'POST' ? {headers: {'Content-Type': 'application/json'}, body: '{}'} : {}),
}).then(async (res) => done({status: res.status, body: await res.json().catch(() => null)}))
  .catch((err) => done({status: 0, error: String(err)}));
"""


def _site_fetch(driver, path: str, method: str = "GET") -> dict:
    try:
        return driver.execute_async_script(_SITE_FETCH_SCRIPT, path, method) or {}
    except Exception as exc:
        return {"status": 0, "error": str(exc)}


def _site_jobs(driver, file_name: str) -> tuple[list[dict] | None, str]:
    """Return this account's server jobs for file_name (newest first), or None with a reason."""
    res = _site_fetch(driver, "/api/jobs")
    body = res.get("body")
    if res.get("status") != 200 or not isinstance(body, list):
        detail = res.get("error") or (body.get("error") if isinstance(body, dict) else "")
        return None, f"HTTP {res.get('status', 0)}" + (f": {detail}" if detail else "")
    return [job for job in body if isinstance(job, dict) and job.get("fileName") == file_name], ""


def _job_status_line(job: dict) -> str:
    line = f"{job.get('status', '?')} {int(job.get('progress') or 0)}% - {job.get('message', '')}".strip()
    if job.get("error"):
        line = f"{line}\n{job['error']}"
    return line


def _site_record_summary(driver, record_id: str) -> tuple[str, str]:
    res = _site_fetch(
        driver,
        f"/api/database/rest/v1/transcriptions?id=eq.{record_id}&select=title,summary:analysis->>summary",
    )
    rows = res.get("body")
    if res.get("status") != 200 or not isinstance(rows, list) or not rows:
        return "", ""
    return str(rows[0].get("title") or "").strip(), str(rows[0].get("summary") or "").strip()


def _upload_card_text(driver, file_name: str) -> str:
    """Text of the Uploads card for file_name; only needed before the server job exists."""
    from selenium.webdriver.common.by import By

    cards = driver.find_elements(
        By.XPATH,
        (
            f"//h4[normalize-space()={_xpath_literal(Path(file_name).name)}]"
            "/ancestor::div[contains(@class,'animate-slideDown')][1]"
        ),
    )
    with contextlib.suppress(Exception):
        return " | ".join(part.strip() for part in (cards[0].text or "").splitlines() if part.strip())
    return ""


def _sign_in_to_bns_site(driver, site_url: str, admin_username: str, admin_password: str) -> str:
    """Open the site and sign in. Returns "" once the admin Upload tab is available, else a reason."""
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    upload_tab = (By.XPATH, "//nav//button[normalize-space()='Upload']")
    signed_in = (By.XPATH, "//nav//button[normalize-space()='Archive']")
    login_user = (By.ID, "bns-username")
    login_pass = (By.ID, "bns-password")
    login_alert = (By.XPATH, "//form//p[@role='alert']")

    def fail(reason: str, label: str) -> str:
        debug_hint = _write_debug_artifacts(driver, label)
        return f"{reason}\n\n{debug_hint}" if debug_hint else reason

    driver.get(site_url)
    try:
        WebDriverWait(driver, 60).until(
            lambda d: d.find_elements(*signed_in)
            or (d.find_elements(*login_user) and d.find_elements(*login_pass))
        )
    except TimeoutException:
        body_text = ""
        with contextlib.suppress(Exception):
            body_text = driver.find_element(By.TAG_NAME, "body").text
        details = _truncate(f"Title: {driver.title or ''}\nURL: {driver.current_url or ''}\n\n{body_text}")
        return fail(f"Sign-in form not found on the BNS site.\n\n{details}", "admin-form-missing")

    if not driver.find_elements(*signed_in):
        user_input = driver.find_element(*login_user)
        pass_input = driver.find_element(*login_pass)
        user_input.clear()
        user_input.send_keys(admin_username)
        pass_input.clear()
        pass_input.send_keys(admin_password)
        submit = driver.find_element(By.XPATH, "//form[.//input[@id='bns-password']]//button[@type='submit']")
        driver.execute_script("arguments[0].click();", submit)
        try:
            WebDriverWait(driver, 30).until(
                lambda d: d.find_elements(*signed_in)
                or any((el.text or "").strip() for el in d.find_elements(*login_alert))
            )
        except TimeoutException:
            return fail("Sign-in did not complete within 30s.", "admin-login-failed")
        if not driver.find_elements(*signed_in):
            alert_text = " ".join((el.text or "").strip() for el in driver.find_elements(*login_alert))
            return fail(f"BNS site rejected the sign-in for {admin_username}: {alert_text}", "admin-login-failed")

    try:
        # The Upload tab only renders for admin accounts, after the session check returns.
        WebDriverWait(driver, 30).until(lambda d: d.find_elements(*upload_tab))
    except TimeoutException:
        return fail(
            f"Signed in as {admin_username}, but the Upload tab is missing. "
            "The account must be listed in the site's ADMIN_CREDENTIALS (not VIEWER_CREDENTIALS).",
            "upload-nav-missing",
        )
    return ""


def upload_to_bns_site(
    file_path: Path,
    site_url: str,
    admin_username: str,
    admin_password: str,
    headless: bool,
    upload_timeout_seconds: int,
    stall_timeout_seconds: int,
) -> dict:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    options = webdriver.ChromeOptions()
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-pipe")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-crash-reporter")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    )
    options.add_argument("--window-size=1600,1400")

    chrome_profile_dir = tempfile.mkdtemp(prefix="bns_chrome_")
    options.add_argument(f"--user-data-dir={chrome_profile_dir}")

    chrome_binary = os.environ.get("BNS_CHROME_BINARY", "").strip()
    if not chrome_binary:
        chrome_binary = (
            shutil.which("chromium-browser")
            or shutil.which("chromium")
            or shutil.which("google-chrome")
            or ""
        )
    if chrome_binary:
        options.binary_location = chrome_binary

    driver_path = os.environ.get("BNS_CHROMEDRIVER_PATH", "").strip()
    if not driver_path:
        driver_path = shutil.which("chromedriver") or ""

    log_path = os.environ.get("BNS_CHROMEDRIVER_LOG_PATH", "").strip()
    log_output = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_output = log_path

    service = Service(executable_path=driver_path, log_output=log_output) if driver_path else Service(log_output=log_output)

    try:
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as exc:
        log_excerpt = _tail_file(Path(log_path)) if log_path else ""
        reason = f"ChromeDriver start failed: {exc}"
        if log_excerpt:
            reason = f"{reason}\n\nChromeDriver log:\n{log_excerpt}"
        return {
            "status": "error",
            "reason": reason,
            "title": "",
            "summary": "",
        }
    def result(status: str, reason: str, title: str = "", summary: str = "") -> dict:
        return {"status": status, "reason": reason, "title": title, "summary": summary}

    try:
        driver.set_page_load_timeout(120)
        driver.set_script_timeout(30)
        sign_in_error = _sign_in_to_bns_site(driver, site_url, admin_username, admin_password)
        if sign_in_error:
            return result("error", sign_in_error)

        file_name = file_path.name

        # Processing runs on the site's server, so a job from an earlier attempt may
        # still be running (or finished) after this script timed out or restarted.
        jobs, jobs_error = _site_jobs(driver, file_name)
        if jobs is None:
            return result("error", f"Could not read the BNS site's upload jobs ({jobs_error}).")
        job = jobs[0] if jobs else None
        if job and job.get("status") in {"error", "stopped"}:
            print(f"Retrying existing BNS website job for {file_name}: {_job_status_line(job)}")
            retry = _site_fetch(driver, f"/api/jobs/{job.get('id')}/retry", "POST")
            if retry.get("status") != 200:
                detail = (retry.get("body") or {}).get("error") or retry.get("error") or ""
                return result(
                    "error",
                    f"Could not retry the failed website job (HTTP {retry.get('status', 0)}: {detail}).\n\n"
                    f"{_truncate(_job_status_line(job), 1800)}",
                )
        elif job:
            print(f"Found existing BNS website job for {file_name}: {_job_status_line(job)}")

        if not job:
            upload_tab = driver.find_element(By.XPATH, "//nav//button[normalize-space()='Upload']")
            driver.execute_script("arguments[0].click();", upload_tab)
            try:
                file_input = WebDriverWait(driver, 60).until(
                    lambda d: (d.find_elements(By.ID, "audio-upload")
                               or d.find_elements(By.CSS_SELECTOR, "input[type='file']") or [None])[0]
                )
            except TimeoutException:
                debug_hint = _write_debug_artifacts(driver, "file-input-missing")
                reason = "Upload file input not found after sign-in."
                return result("error", f"{reason}\n\n{debug_hint}" if debug_hint else reason)
            file_input.send_keys(str(file_path.resolve()))

            try:
                alert = WebDriverWait(driver, 3).until(EC.alert_is_present())
                alert_text = (alert.text or "").strip()
                if "already uploaded" in alert_text.lower():
                    alert.dismiss()
                    return result("skipped_duplicate", alert_text or "Already uploaded alert detected")
                alert.accept()
            except TimeoutException:
                pass

        deadline = time.time() + max(60, upload_timeout_seconds)
        last_status = ""
        last_change_at = time.time()
        next_progress_log_at = time.time() + 60
        job_seen = bool(job)
        while time.time() < deadline:
            jobs, jobs_error = _site_jobs(driver, file_name)
            job = jobs[0] if jobs else None
            if job:
                job_seen = True
                status = _job_status_line(job)
                if job.get("status") == "completed":
                    title, summary = _site_record_summary(driver, job.get("recordId") or job.get("id") or "")
                    return result("uploaded", job.get("message") or "Completed", title, summary)
                if job.get("status") in {"error", "stopped"}:
                    return result("error", _truncate(status, 1800) or "Website upload failed.")
            else:
                # Until the audio reaches storage the job exists only in the page.
                status = _upload_card_text(driver, file_name)
                if "could not be queued" in status.lower():
                    browser_excerpt = _browser_error_excerpt(driver)
                    reason = _truncate(status, 1800)
                    if browser_excerpt:
                        reason = f"{reason}\n\nBrowser console:\n{browser_excerpt}"
                    return result("error", reason)
            if status and status != last_status:
                last_status = status
                last_change_at = time.time()
            if time.time() >= next_progress_log_at:
                shown = _truncate(status, 500) or f"waiting for upload to start ({jobs_error or 'no job yet'})"
                print(f"BNS website upload still running for {file_name}: {shown}")
                next_progress_log_at = time.time() + 60
            if not job_seen and time.time() - last_change_at >= max(20, stall_timeout_seconds):
                reason = (
                    f"Audio upload made no progress for {int(time.time() - last_change_at)}s "
                    "and never reached the site's job queue."
                )
                debug_hint = _write_debug_artifacts(driver, "upload-stalled")
                if debug_hint:
                    reason = f"{reason}\n\n{debug_hint}"
                if last_status:
                    reason = f"{reason}\n\nLast status:\n{_truncate(last_status, 1800)}"
                browser_excerpt = _browser_error_excerpt(driver)
                if browser_excerpt:
                    reason = f"{reason}\n\nBrowser console:\n{browser_excerpt}"
                return result("error", reason)
            time.sleep(5 if job_seen else 2)

        reason = f"Timed out waiting for upload completion after {upload_timeout_seconds}s"
        if job_seen:
            reason = f"{reason}. The job keeps running on the site; the next run will pick it up without re-uploading."
        debug_hint = _write_debug_artifacts(driver, "upload-timeout")
        if debug_hint:
            reason = f"{reason}\n\n{debug_hint}"
        if last_status:
            reason = f"{reason}\n\nLast job status:\n{_truncate(last_status, 1800)}"
        return result("error", reason)
    finally:
        driver.quit()
        with contextlib.suppress(Exception):
            shutil.rmtree(chrome_profile_dir, ignore_errors=True)


def _ext_from_media(msg) -> str:
    """Try to determine a sensible file extension for a message's media."""
    # Document (includes videos/voice/stickers sent as "file")
    if getattr(msg, "document", None):
        try:
            ext = tutils.get_extension(msg.document)
            if ext:
                return ext
        except Exception:
            pass
        mime = getattr(msg.document, "mime_type", None)
        return MIME_EXT_FALLBACK.get(mime, ".bin")
    # Photo
    if getattr(msg, "photo", None):
        return ".jpg"
    return ".bin"


def is_audio_message(msg) -> bool:
    """Return True if the message contains audio (including voice)."""
    if getattr(msg, "voice", None) or getattr(msg, "audio", None):
        return True
    doc = getattr(msg, "document", None)
    if doc and getattr(doc, "mime_type", "").startswith("audio/"):
        return True
    for attr in getattr(doc, "attributes", []) or []:
        if isinstance(attr, types.DocumentAttributeAudio):
            return True
    return False


def get_audio_duration_seconds(input_path: Path) -> float | None:
    if shutil.which(FFPROBE_PATH) is None:
        return None
    proc = subprocess.run(
        [
            FFPROBE_PATH,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    try:
        duration = float((proc.stdout or "").strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


# Use encoder-supported rates and leave room for headers, tags, and frame padding.
MP3_BITRATES_KBPS = (192, 160, 128, 112, 96, 80, 64, 56, 48, 40, 32, 24, 16, 8)
MP3_UPLOAD_HEADROOM = 0.97


def mp3_bitrate_for_upload(input_path: Path) -> int | None:
    duration = get_audio_duration_seconds(input_path)
    if not duration:
        return None
    target_kbps = BNS_MAX_UPLOAD_BYTES * MP3_UPLOAD_HEADROOM * 8 / duration / 1000
    return next((rate for rate in MP3_BITRATES_KBPS if rate <= target_kbps), 8)


def convert_audio_to_mp3(input_path: Path) -> Path:
    """Convert audio to an MP3 that fits the measured upload limit."""
    if input_path.suffix.lower() == ".mp3" and input_path.stat().st_size <= BNS_MAX_UPLOAD_BYTES:
        return input_path
    if shutil.which(FFMPEG_PATH) is None:
        raise RuntimeError(f"ffmpeg not found: {FFMPEG_PATH}")

    output_path = input_path.with_suffix(".mp3")
    if output_path == input_path:
        output_path = input_path.with_name(f"{input_path.stem}-compressed.mp3")
    if output_path.exists():
        output_path.unlink()

    bitrate = mp3_bitrate_for_upload(input_path)
    # Bound conversion work even when ffprobe cannot determine duration.
    for attempt in range(4):
        audio_options = ["-b:a", f"{bitrate}k"] if bitrate else ["-q:a", "2"]
        cmd = [
            FFMPEG_PATH, "-y", "-i", str(input_path), "-vn",
            "-c:a", "libmp3lame", *audio_options, str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
            raise RuntimeError(detail)
        size = output_path.stat().st_size
        if size <= BNS_MAX_UPLOAD_BYTES:
            return output_path
        if bitrate is None:
            duration = get_audio_duration_seconds(output_path)
            estimated_rate = size * 8 / duration / 1000 if duration else 192
        else:
            estimated_rate = bitrate
        target_rate = estimated_rate * BNS_MAX_UPLOAD_BYTES / size * MP3_UPLOAD_HEADROOM
        lower_rates = [rate for rate in MP3_BITRATES_KBPS
                       if rate < estimated_rate and rate <= target_rate]
        if not lower_rates or attempt == 3:
            break
        bitrate = lower_rates[0]
        print(f"MP3 exceeds upload limit ({size:,} bytes); retrying at {bitrate} kbps", flush=True)
    output_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"MP3 is still too large: {size / 1024 / 1024:.3f} MiB ({size:,} bytes) "
        f"(limit {BNS_MAX_UPLOAD_BYTES / 1024 / 1024:.3f} MiB; {BNS_MAX_UPLOAD_BYTES:,} bytes)"
    )


async def resolve_entity_safely(client: TelegramClient, ref):
    """Given an int id or username, try to return an input entity usable by methods."""
    await client.get_dialogs(limit=None)
    try:
        return await client.get_input_entity(ref)
    except Exception:
        # If ref is an integer channel id (e.g., 1773081661) Telethon often needs PeerChannel(abs(id))
        if isinstance(ref, int):
            return await client.get_input_entity(types.PeerChannel(abs(ref)))
        raise


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=DEFAULT_SOURCE, help="Source channel id (numeric)")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Target channel id (numeric)")
    ap.add_argument("--outdir", default=str(Path("downloads") / "Brahma naa sange"))
    ap.add_argument(
        "--seed",
        action="store_true",
        help="If set and no state exists, process recent messages instead of seeding state.",
    )
    ap.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help="State file path for last processed id.",
    )
    ap.add_argument("--audio-only", action="store_true", help="Only process and send audio messages.")
    ap.add_argument(
        "--session",
        default="",
        help="Override TELEGRAM_SESSION path for this run.",
    )
    ap.add_argument(
        "--site-upload-timeout",
        type=int,
        default=int(os.environ.get("BNS_UPLOAD_TIMEOUT_SECONDS", "2400")),
        help="Max seconds to wait for website upload completion per file.",
    )
    ap.add_argument(
        "--site-upload-stall-timeout",
        type=int,
        default=int(os.environ.get("BNS_UPLOAD_STALL_SECONDS", "90")),
        help="Seconds the audio upload may show no progress before it is reported as stalled.",
    )
    ap.add_argument(
        "--no-site-upload",
        action="store_true",
        help="Disable Selenium website upload step for this run.",
    )
    ap.add_argument(
        "--only-message-id",
        type=int,
        default=0,
        help="Process only this source message id (must be newer than the checkpoint).",
    )
    ap.add_argument("--stop-after-today", action="store_true",
                    help="Stop after completing a source post dated today (used by the recovery worker).")
    args = ap.parse_args()
    with exclusive_lock(str(Path(args.state_file).resolve()) + ".lock"):
        await run_sync(args)


async def run_sync(args):

    api_id = int(os.environ.get("TELEGRAM_API_ID", "0"))
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    session = args.session or os.environ.get("TELEGRAM_SESSION", "media_grabber_session.session")
    bot_token = os.environ.get("BOT_TOKEN", "")
    notify_chat_id = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "")
    site_url = os.environ.get("BNS_SITE_URL", "https://bns.classchats.net")
    admin_username, admin_password = _resolve_bns_admin_credentials()
    selenium_headless = _env_bool("BNS_SELENIUM_HEADLESS", True)
    skip_site_upload = args.no_site_upload or _env_bool("BNS_DISABLE_SITE_UPLOAD", False)

    if not api_id or not api_hash:
        raise RuntimeError("TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables are required")

    site_upload_run_allowed = args.audio_only
    can_site_upload = (
        (not skip_site_upload)
        and site_upload_run_allowed
        and bool(admin_username)
        and bool(admin_password)
    )
    if not site_upload_run_allowed and not skip_site_upload:
        print("Website upload disabled: only --audio-only runs may upload to the BNS site")
    elif args.audio_only and not can_site_upload and not skip_site_upload:
        raise RuntimeError("Website upload requires BNS_ADMIN_USERNAME and BNS_ADMIN_PASSWORD; explicitly disable it to skip")
    if not bot_token or not notify_chat_id:
        print("Telegram notifications disabled: set BOT_TOKEN and TELEGRAM_NOTIFY_CHAT_ID in .env")

    client = TelegramClient(
        load_download_session(session),
        api_id,
        api_hash,
        timeout=60,
        request_retries=5,
        connection_retries=5,
    )

    state_file = Path(args.state_file)
    state = load_state(state_file)
    last_id = int(state.get("last_id", 0))
    today = datetime.now(ZoneInfo(os.environ.get("BNS_TIMEZONE", "America/New_York"))).date().isoformat()
    stop_after_today = getattr(args, "stop_after_today", False)
    if stop_after_today and state.get("last_delivery_date", "") >= today:
        print("Today's BNS is already delivered.")
        return

    try:
        await client.connect()
        if not await client.is_user_authorized():
            raise RuntimeError("Telegram session is not authorized; run downloader.py --login on this host")
        print("Logged in")

        src_ent = await resolve_entity_safely(client, args.source)
        tgt_ent = await resolve_entity_safely(client, args.target)

        # Fetch oldest unseen messages, so an outage longer than 200 posts loses nothing.
        if args.only_message_id:
            if args.only_message_id <= last_id:
                raise RuntimeError("Requested message is already processed; restore an appropriate checkpoint first")
            msg = await client.get_messages(src_ent, ids=args.only_message_id)
            if not msg:
                raise RuntimeError("Requested source message was not found")
            to_process = [msg]
        elif last_id == 0:
            recent = await client.get_messages(src_ent, limit=200 if args.seed else 1)
            if not recent:
                print("No messages found in source channel")
                return
            if not args.seed:
                state["last_id"] = recent[0].id
                save_state(state_file, state)
                print(f"No state found. Seeded last_id={recent[0].id}; no historical deliveries made.")
                return
            to_process = sorted(recent, key=lambda msg: msg.id)
        else:
            to_process = [msg async for msg in client.iter_messages(
                src_ent, min_id=last_id, reverse=True, limit=200,
            )]
        if not to_process:
            print("No new messages to process.")
            return

        for msg in to_process:
            tempd = None
            try:
                if not msg.media:
                    print(f"Skipping {msg.id}: no media")
                    state["last_id"] = msg.id
                    save_state(state_file, state)
                    continue

                if args.audio_only and not is_audio_message(msg):
                    print(f"Skipping {msg.id}: not audio")
                    state["last_id"] = msg.id
                    save_state(state_file, state)
                    continue

                print(f"Processing message {msg.id}...")
                # Download to temporary file (we'll rename it to date-based filename)
                tempd = tempfile.mkdtemp(prefix="bns_")
                path = await asyncio.wait_for(msg.download_media(file=tempd), timeout=1800)
                if not path:
                    raise RuntimeError(f"Failed to download media for {msg.id}; retained for retry")

                # Build date-based filename (use message date if available)
                msg_date = getattr(msg, "date", None)
                if msg_date is not None:
                    date_str = msg_date.date().isoformat()
                else:
                    from datetime import date

                    date_str = date.today().isoformat()

                ext = _ext_from_media(msg)
                base_name = f"{date_str}{ext}"
                final_path = Path(tempd) / base_name

                # move downloaded file to final_path
                try:
                    source_path = Path(path)
                    if source_path.resolve() != final_path.resolve():
                        if final_path.exists():
                            final_path.unlink()
                        source_path.rename(final_path)
                    path = str(final_path)
                except Exception:
                    # fallback: copy
                    shutil.copy(path, str(final_path))
                    path = str(final_path)

                caption = None
                # Telethon stores message text in .message
                if getattr(msg, "message", None):
                    caption = msg.message

                if args.audio_only and Path(path).suffix.lower() in AUDIO_EXTS:
                    try:
                        upload_path = await asyncio.to_thread(convert_audio_to_mp3, Path(path))
                        if upload_path != Path(path):
                            print(f"Prepared MP3 under {BNS_MAX_UPLOAD_MB:.1f} MB: {upload_path.name}")
                        path = str(upload_path)
                    except Exception as conv_exc:
                        notify(
                            bot_token,
                            notify_chat_id,
                            "\n".join(
                                [
                                    "[ERROR] BNS Audio MP3 Conversion Failed",
                                    f"File: {Path(path).name}",
                                    f"Error: {conv_exc}",
                                ]
                            ),
                        )
                        raise

                steps = state.setdefault("pending", {}).setdefault(str(msg.id), {})
                if can_site_upload and not steps.get("website"):
                    print(f"Uploading {Path(path).name} to {site_url} via Selenium...")
                    site_result = await asyncio.to_thread(
                        upload_to_bns_site,
                        file_path=Path(path), site_url=site_url,
                        admin_username=admin_username, admin_password=admin_password,
                        headless=selenium_headless,
                        upload_timeout_seconds=args.site_upload_timeout,
                        stall_timeout_seconds=args.site_upload_stall_timeout,
                    )
                    if site_result.get("status") not in {"uploaded", "skipped_duplicate"}:
                        raise RuntimeError(site_result.get("reason") or "Website upload failed")
                    steps["website"] = True
                    save_state(state_file, state)
                    notice = [f"[OK] BNS website: {Path(path).name} ({site_result['status']})"]
                    for key in ("reason", "title", "summary"):
                        if site_result.get(key):
                            notice.append(f"{key.title()}: {_truncate(str(site_result[key]), 1200)}")
                    notify(bot_token, notify_chat_id, "\n".join(notice))

                # Re-upload to target channel using the date-named file
                print(
                    f"Uploading to target channel {args.target} with filename {Path(path).name}..."
                )
                if not steps.get("telegram"):
                    await asyncio.wait_for(client.send_file(tgt_ent, path, caption=caption), timeout=1800)
                    steps["telegram"] = True
                    save_state(state_file, state)
                print(f"Uploaded message {msg.id} -> target")

                # Advance only after every required destination succeeds.
                state["last_id"] = msg.id
                state["pending"].pop(str(msg.id), None)
                state["last_success_at"] = time.time()
                delivery_date = (msg_date.astimezone(ZoneInfo(os.environ.get("BNS_TIMEZONE", "America/New_York")))
                                 .date().isoformat()) if msg_date else date_str
                state["last_delivery_date"] = max(state.get("last_delivery_date", ""), delivery_date)
                save_state(state_file, state)
                if stop_after_today and delivery_date >= today:
                    print("Today's BNS delivered; stopping until tomorrow.")
                    break

            except Exception as exc:
                # Stop here: later successes must never overtake this failed message.
                print(f"Error processing message {msg.id}; will retry: {exc}")
                raise
            finally:
                if tempd:
                    shutil.rmtree(tempd, ignore_errors=True)

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
