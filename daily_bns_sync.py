#!/usr/bin/env python3
"""
daily_bns_sync.py

Fetch new media posts from a source channel and upload them to a target channel.

Default source: 1773081661
Default target: 3130614830

Behaviour:
- Maintains a small state file at .state/bns_state.json with last_processed_id.
- On first run (no state), seeds the last_processed_id to the latest message and exits to avoid backfilling.
  Use --seed to force processing of the recent window.

Requires TELEGRAM_API_ID, TELEGRAM_API_HASH and TELEGRAM_SESSION env variables (session file path).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from telethon import TelegramClient, errors, types, utils as tutils  # type: ignore

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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def _extract_recent_upload_summary(driver) -> tuple[str, str]:
    from selenium.webdriver.common.by import By

    recent_sections = driver.find_elements(
        By.XPATH,
        "//h3[normalize-space()='Recent uploads']/ancestor::div[contains(@class,'max-w-5xl')][1]",
    )
    if not recent_sections:
        return "", ""
    section = recent_sections[0]
    titles = section.find_elements(By.XPATH, ".//h4")
    summaries = section.find_elements(
        By.XPATH, ".//p[contains(@class,'text-sm') and contains(@class,'text-gray-600')]"
    )
    title = titles[0].text.strip() if titles else ""
    summary = summaries[0].text.strip() if summaries else ""
    return title, summary


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


def _expected_upload_title(file_name: str) -> str:
    stem = Path(file_name).stem
    base = stem.split(" (", 1)[0]
    try:
        dt = datetime.strptime(base, "%Y-%m-%d")
    except ValueError:
        return ""

    day = dt.day
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return dt.strftime(f"%B {day}{suffix}, %Y")


def _page_has_uploaded_title(driver, expected_title: str) -> bool:
    if not expected_title:
        return False

    from selenium.webdriver.common.by import By

    matches = driver.find_elements(
        By.XPATH,
        f"//h4[normalize-space()={_xpath_literal(expected_title)}]",
    )
    return bool(matches)


def _upload_calendar_day_uploaded(driver, file_name: str) -> bool:
    stem = Path(file_name).stem
    base = stem.split(" (", 1)[0]
    try:
        dt = datetime.strptime(base, "%Y-%m-%d")
    except ValueError:
        return False

    from selenium.webdriver.common.by import By

    month_label = dt.strftime("%B %Y")
    day_label = str(dt.day)
    buttons = driver.find_elements(
        By.XPATH,
        (
            "//h3[normalize-space()='Upload calendar']"
            "/ancestor::div[contains(@class,'max-w-5xl')][1]"
            f"//span[normalize-space()={_xpath_literal(month_label)}]"
            "/ancestor::div[contains(@class,'bg-white')][1]"
            f"//button[.//span[normalize-space()={_xpath_literal(day_label)}]]"
        ),
    )
    for button in buttons:
        classes = button.get_attribute("class") or ""
        if "border-orange-200" in classes or "bg-orange-50" in classes:
            return True
        markers = button.find_elements(
            By.XPATH, ".//span[contains(@class,'bg-orange-500')]"
        )
        if markers:
            return True
    return False


def _page_has_completed_queue_item(driver, file_name: str, body_text: str) -> bool:
    compact_text = "".join((body_text or "").upper().split())
    stem = Path(file_name).stem.upper()
    name = file_name.upper()
    if "PROCESSINGQUEUE" not in compact_text:
        return False
    if stem not in compact_text and name not in compact_text:
        return False

    from selenium.webdriver.common.by import By

    for node in driver.find_elements(
        By.XPATH,
        f"//*[contains(normalize-space(), {_xpath_literal(Path(file_name).name)})]",
    ):
        with contextlib.suppress(Exception):
            card = node.find_element(
                By.XPATH,
                "./ancestor::div[contains(@class,'border') or contains(@class,'rounded')][1]",
            )
            card_text = "".join((card.text or "").upper().split())
            if "FAILED" in card_text or "ERROR" in card_text:
                return False
            if "ARCHIVED" in card_text or ("COMPLETED" in card_text and "100%" in card_text):
                return True

    return (
        "ARCHIVED" in compact_text
        and "FAILED" not in compact_text
        and "ERROR" not in compact_text
    )


def _extract_card_error(card, browser_excerpt: str) -> str:
    from selenium.webdriver.common.by import By

    err_nodes = card.find_elements(
        By.XPATH,
        ".//*[contains(@class,'text-red-600') or contains(@class,'text-red-500') or contains(@class,'text-red-700')]",
    )
    err_text = "\n".join(
        node.text.strip() for node in err_nodes if node.text and node.text.strip()
    ).strip()
    if err_text:
        return _truncate(err_text, 1800)

    card_text = card.text.strip()
    if card_text:
        return _truncate(card_text, 1800)

    return browser_excerpt or "Website upload failed, but no visible error text was found."


def upload_to_bns_site(
    file_path: Path,
    site_url: str,
    admin_username: str,
    admin_password: str,
    headless: bool,
    upload_timeout_seconds: int,
    stall_timeout_seconds: int,
) -> dict:
    import shutil

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
    try:
        driver.set_page_load_timeout(120)
        driver.get(site_url)
        wait = WebDriverWait(driver, 30)

        def admin_ready(d) -> bool:
            return bool(
                d.find_elements(By.XPATH, "//*[contains(normalize-space(),'Admin:')]")
                or d.find_elements(By.XPATH, "//button[normalize-space()='Upload']")
                or d.find_elements(By.XPATH, "//button[contains(normalize-space(),'Add Recording')]")
            )

        def reveal_admin_form() -> None:
            if admin_ready(driver):
                return
            try:
                WebDriverWait(driver, 30).until(
                    lambda d: admin_ready(d)
                    or d.find_elements(
                        By.XPATH,
                        (
                            "//button[normalize-space()='Are you an admin?']"
                            "|//button[contains(translate(normalize-space(),"
                            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'admin')]"
                        ),
                    )
                    or (
                        d.find_elements(By.XPATH, "//input[@placeholder='Admin username']")
                        and d.find_elements(By.XPATH, "//input[@placeholder='Admin password']")
                    )
                )
            except TimeoutException:
                return

            for _ in range(6):
                if admin_ready(driver):
                    return
                buttons = driver.find_elements(
                    By.XPATH,
                    (
                        "//button[normalize-space()='Are you an admin?']"
                        "|//button[contains(translate(normalize-space(),"
                        "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'admin')]"
                    ),
                )
                clicked = False
                for button in reversed(buttons):
                    try:
                        driver.execute_script("arguments[0].click();", button)
                        clicked = True
                        break
                    except Exception:
                        continue
                if not clicked:
                    time.sleep(1)
                    continue
                try:
                    WebDriverWait(driver, 10).until(
                        lambda d: admin_ready(d)
                        or (
                            d.find_elements(By.XPATH, "//input[@placeholder='Admin username']")
                            and d.find_elements(By.XPATH, "//input[@placeholder='Admin password']")
                        )
                    )
                    return
                except TimeoutException:
                    continue

        reveal_admin_form()

        already_admin = driver.find_elements(
            By.XPATH, "//*[contains(normalize-space(),'Admin:')]"
        )
        if not already_admin:
            try:
                wait.until(
                    lambda d: (
                        d.find_elements(By.XPATH, "//input[@placeholder='Admin username']")
                        and d.find_elements(By.XPATH, "//input[@placeholder='Admin password']")
                    )
                )
                user_input = driver.find_element(
                    By.XPATH, "//input[@placeholder='Admin username']"
                )
                pass_input = driver.find_element(
                    By.XPATH, "//input[@placeholder='Admin password']"
                )
            except TimeoutException:
                page_title = driver.title or ""
                page_url = driver.current_url or ""
                body_text = ""
                with contextlib.suppress(Exception):
                    body_text = driver.find_element(By.TAG_NAME, "body").text
                debug_hint = _write_debug_artifacts(driver, "admin-form-missing")
                reason = "Admin login form not found on the upload page."
                details = _truncate(f"Title: {page_title}\nURL: {page_url}\n\n{body_text}")
                if debug_hint:
                    details = f"{details}\n\n{debug_hint}"
                return {
                    "status": "error",
                    "reason": f"{reason}\n\n{details}",
                    "title": "",
                    "summary": "",
                }
            user_input.clear()
            user_input.send_keys(admin_username)
            pass_input.clear()
            pass_input.send_keys(admin_password)

            verify_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='Verify']"))
            )
            driver.execute_script("arguments[0].click();", verify_btn)

            try:
                WebDriverWait(driver, 20).until(admin_ready)
            except TimeoutException:
                debug_hint = _write_debug_artifacts(driver, "admin-login-failed")
                reason = "Admin verification did not unlock the upload controls."
                if debug_hint:
                    reason = f"{reason}\n\n{debug_hint}"
                return {
                    "status": "error",
                    "reason": reason,
                    "title": "",
                    "summary": "",
                }

        if not admin_ready(driver):
            reveal_admin_form()
            wait.until(admin_ready)

        nav_clicked = bool(
            driver.find_elements(By.ID, "audio-upload")
            or driver.find_elements(By.XPATH, "//h2[normalize-space()='Upload Recording']")
        )

        if not nav_clicked:
            upload_locators = [
                (By.XPATH, "//button[normalize-space()='Upload']"),
                (By.XPATH, "//button[contains(normalize-space(),'Add Recording')]")
            ]
            for locator in upload_locators:
                try:
                    upload_btn = WebDriverWait(driver, 15).until(
                        EC.element_to_be_clickable(locator)
                    )
                    driver.execute_script("arguments[0].click();", upload_btn)
                    nav_clicked = True
                    break
                except TimeoutException:
                    continue

        if not nav_clicked:
            debug_hint = _write_debug_artifacts(driver, "upload-nav-missing")
            reason = "Upload navigation button not found after admin login."
            if debug_hint:
                reason = f"{reason}\n\n{debug_hint}"
            return {
                "status": "error",
                "reason": reason,
                "title": "",
                "summary": "",
            }

        file_wait = WebDriverWait(driver, 60)
        file_wait.until(
            lambda d: d.find_elements(By.ID, "audio-upload")
            or d.find_elements(By.XPATH, "//h2[normalize-space()='Upload Recording']")
            or admin_ready(d)
        )

        if (
            not driver.find_elements(By.ID, "audio-upload")
            and not driver.find_elements(By.XPATH, "//h2[normalize-space()='Upload Recording']")
            and admin_ready(driver)
        ):
            for locator in (
                (By.XPATH, "//button[normalize-space()='Upload']"),
                (By.XPATH, "//button[contains(normalize-space(),'Add Recording')]"),
            ):
                with contextlib.suppress(Exception):
                    upload_btn = driver.find_element(*locator)
                    driver.execute_script("arguments[0].click();", upload_btn)
                    break

        try:
            file_wait.until(
                lambda d: d.find_elements(By.ID, "audio-upload")
                or d.find_elements(By.XPATH, "//h2[normalize-space()='Upload Recording']")
            )
        except TimeoutException:
            pass

        file_input = None
        for locator in (
            (By.ID, "audio-upload"),
            (By.CSS_SELECTOR, "input[type='file']"),
        ):
            try:
                file_input = file_wait.until(EC.presence_of_element_located(locator))
                if file_input:
                    break
            except TimeoutException:
                continue

        if not file_input:
            debug_hint = _write_debug_artifacts(driver, "file-input-missing")
            reason = "Upload file input not found after admin login."
            if debug_hint:
                reason = f"{reason}\n\n{debug_hint}"
            return {
                "status": "error",
                "reason": reason,
                "title": "",
                "summary": "",
            }
        file_input.send_keys(str(file_path.resolve()))

        try:
            alert = WebDriverWait(driver, 3).until(EC.alert_is_present())
            alert_text = (alert.text or "").strip()
            if "already uploaded" in alert_text.lower():
                alert.dismiss()
                return {
                    "status": "skipped_duplicate",
                    "reason": alert_text or "Already uploaded alert detected",
                    "title": "",
                    "summary": "",
                }
            alert.accept()
        except TimeoutException:
            pass

        file_name = file_path.name
        expected_title = _expected_upload_title(file_name)
        deadline = time.time() + max(60, upload_timeout_seconds)
        last_card_text = ""
        last_change_at = time.time()
        while time.time() < deadline:
            browser_excerpt = _browser_error_excerpt(driver)
            body_text = ""
            with contextlib.suppress(Exception):
                body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
            if body_text and body_text != last_card_text:
                last_card_text = body_text
                last_change_at = time.time()

            compact_text = "".join(body_text.upper().split())
            if (
                "SUCCESSFULLYARCHIVED" in compact_text
                or "ARCHIVED(RATELIMITSDETECTED/RECOVERED)" in compact_text
                or _page_has_completed_queue_item(driver, file_name, body_text)
            ):
                title, summary = _extract_recent_upload_summary(driver)
                return {
                    "status": "uploaded",
                    "reason": "Completed",
                    "title": title,
                    "summary": summary,
                }
            if expected_title and _page_has_uploaded_title(driver, expected_title):
                title, summary = _extract_recent_upload_summary(driver)
                return {
                    "status": "uploaded",
                    "reason": f"Found expected uploaded title {expected_title}",
                    "title": title or expected_title,
                    "summary": summary,
                }
            if _upload_calendar_day_uploaded(driver, file_name):
                title, summary = _extract_recent_upload_summary(driver)
                return {
                    "status": "uploaded",
                    "reason": f"Upload calendar now marks {file_name}",
                    "title": title or expected_title,
                    "summary": summary,
                }
            if (
                "ERROR" in compact_text
                or "PROCESSINGFAILED" in compact_text
                or "RESOURCE_EXHAUSTED" in compact_text
            ):
                err_text = _truncate(body_text, 1800) or "Website upload failed."
                if browser_excerpt and browser_excerpt not in err_text:
                    err_text = f"{err_text}\n\nBrowser console:\n{browser_excerpt}"
                return {
                    "status": "error",
                    "reason": err_text,
                    "title": "",
                    "summary": "",
                }
            if browser_excerpt and any(
                token in browser_excerpt.lower()
                for token in ("429", "quota", "resource_exhausted")
            ):
                if time.time() - last_change_at >= max(20, stall_timeout_seconds):
                    return {
                        "status": "error",
                        "reason": (
                            "Website upload appears stalled after a browser/API quota error.\n\n"
                            f"Browser console:\n{browser_excerpt}"
                        ),
                        "title": "",
                        "summary": "",
                    }
            time.sleep(2)

        with contextlib.suppress(Exception):
            body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
            if _page_has_completed_queue_item(driver, file_name, body_text):
                title, summary = _extract_recent_upload_summary(driver)
                return {
                    "status": "uploaded",
                    "reason": "Completed after final status check",
                    "title": title,
                    "summary": summary,
                }

        browser_excerpt = _browser_error_excerpt(driver)
        reason = f"Timed out waiting for upload completion after {upload_timeout_seconds}s"
        debug_hint = _write_debug_artifacts(driver, "upload-timeout")
        if debug_hint:
            reason = f"{reason}\n\n{debug_hint}"
        if browser_excerpt:
            reason = f"{reason}\n\nBrowser console:\n{browser_excerpt}"
        return {
            "status": "error",
            "reason": reason,
            "title": "",
            "summary": "",
        }
    finally:
        driver.quit()
        with contextlib.suppress(Exception):
            import shutil

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
    )
    if proc.returncode != 0:
        return None
    try:
        duration = float((proc.stdout or "").strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def mp3_bitrate_for_upload(input_path: Path) -> int | None:
    duration = get_audio_duration_seconds(input_path)
    if not duration:
        return None
    target_kbps = int((BNS_MAX_UPLOAD_BYTES * 8) / duration / 1000)
    return max(64, min(192, target_kbps))


def convert_audio_to_mp3(input_path: Path) -> Path:
    """Convert audio to a high-quality MP3 for Telegram channel delivery."""
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
    audio_options = ["-b:a", f"{bitrate}k"] if bitrate else ["-q:a", "2"]
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-c:a",
        "libmp3lame",
        *audio_options,
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
        raise RuntimeError(detail)
    if output_path.stat().st_size > BNS_MAX_UPLOAD_BYTES:
        raise RuntimeError(
            f"MP3 is still too large: {output_path.stat().st_size / 1024 / 1024:.1f} MB "
            f"(limit {BNS_MAX_UPLOAD_MB:.1f} MB)"
        )
    return output_path


def load_state(state_file: Path) -> dict:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text())
    except Exception:
        return {}


def save_state(state_file: Path, state: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))


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
        help="Seconds with no visible upload progress before surfacing browser quota errors.",
    )
    ap.add_argument(
        "--no-site-upload",
        action="store_true",
        help="Disable Selenium website upload step for this run.",
    )
    args = ap.parse_args()

    api_id = int(os.environ.get("TELEGRAM_API_ID", "0"))
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    session = args.session or os.environ.get("TELEGRAM_SESSION", "media_grabber_session.session")
    bot_token = os.environ.get("BOT_TOKEN", "")
    notify_chat_id = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "")
    site_url = os.environ.get("BNS_SITE_URL", "https://bns.classchats.net")
    admin_username = os.environ.get("BNS_ADMIN_USERNAME", "").strip()
    admin_password = os.environ.get("BNS_ADMIN_PASSWORD", "").strip()
    selenium_headless = _env_bool("BNS_SELENIUM_HEADLESS", True)
    skip_site_upload = args.no_site_upload or _env_bool("BNS_DISABLE_SITE_UPLOAD", False)

    if not api_id or not api_hash:
        print("TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables are required")
        return

    site_upload_run_allowed = args.audio_only
    can_site_upload = (
        (not skip_site_upload)
        and site_upload_run_allowed
        and bool(admin_username)
        and bool(admin_password)
    )
    if not site_upload_run_allowed and not skip_site_upload:
        print("Website upload disabled: only --audio-only runs may upload to the BNS site")
    elif not can_site_upload and not skip_site_upload:
        print("Website upload disabled: set BNS_ADMIN_USERNAME and BNS_ADMIN_PASSWORD in .env")
    if not bot_token or not notify_chat_id:
        print("Telegram notifications disabled: set BOT_TOKEN and TELEGRAM_NOTIFY_CHAT_ID in .env")

    client = TelegramClient(
        session,
        api_id,
        api_hash,
        timeout=60,
        request_retries=5,
        connection_retries=5,
    )

    state_file = Path(args.state_file)
    state = load_state(state_file)
    last_id = int(state.get("last_id", 0))

    try:
        await client.start()
        print("Logged in")

        src_ent = await resolve_entity_safely(client, args.source)
        tgt_ent = await resolve_entity_safely(client, args.target)

        # Fetch recent messages (window) and decide what to process
        recent = await client.get_messages(src_ent, limit=200)
        if not recent:
            print("No messages found in source channel")
            return

        # Determine seed behaviour
        if last_id == 0:
            if not args.seed:
                # Seed latest and exit
                newest = recent[0]
                state["last_id"] = newest.id
                save_state(state_file, state)
                print(
                    f"No state found. Seeded last_id={newest.id}. Use --seed to process recent posts."
                )
                return
            print("No state found but --seed specified: will process recent window.")

        # Gather messages newer than last_id
        to_process = [m for m in recent if m.id > last_id]
        if not to_process:
            print("No new messages to process.")
            return

        # Process oldest -> newest
        to_process.sort(key=lambda m: m.id)

        for msg in to_process:
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
                path = await msg.download_media(file=tempd)
                if not path:
                    print(f"Failed to download media for {msg.id}, skipping")
                    state["last_id"] = msg.id
                    save_state(state_file, state)
                    continue

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
                    import shutil

                    shutil.copy(path, str(final_path))
                    path = str(final_path)

                caption = None
                # Telethon stores message text in .message
                if getattr(msg, "message", None):
                    caption = msg.message

                if args.audio_only and Path(path).suffix.lower() in AUDIO_EXTS:
                    try:
                        upload_path = convert_audio_to_mp3(Path(path))
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

                if can_site_upload:
                    try:
                        print(f"Uploading {Path(path).name} to {site_url} via Selenium...")
                        site_result = upload_to_bns_site(
                            file_path=Path(path),
                            site_url=site_url,
                            admin_username=admin_username,
                            admin_password=admin_password,
                            headless=selenium_headless,
                            upload_timeout_seconds=args.site_upload_timeout,
                            stall_timeout_seconds=args.site_upload_stall_timeout,
                        )
                        status = site_result.get("status", "unknown")
                        title = (site_result.get("title") or "").strip()
                        summary = (site_result.get("summary") or "").strip()
                        reason = (site_result.get("reason") or "").strip()

                        if status == "uploaded":
                            notify_lines = [
                                "[OK] BNS Website Upload Success",
                                f"File: {Path(path).name}",
                            ]
                            if title:
                                notify_lines.append(f"Title: {title}")
                            if summary:
                                notify_lines.append(f"Summary: {_truncate(summary, 1200)}")
                            notify(bot_token, notify_chat_id, "\n".join(notify_lines))
                        elif status == "skipped_duplicate":
                            notify(
                                bot_token,
                                notify_chat_id,
                                "\n".join(
                                    [
                                        "[SKIP] BNS Website Upload Skipped (Already Uploaded)",
                                        f"File: {Path(path).name}",
                                        f"Reason: {reason or 'Duplicate detected'}",
                                    ]
                                ),
                            )
                        else:
                            notify(
                                bot_token,
                                notify_chat_id,
                                "\n".join(
                                    [
                                        "[ERROR] BNS Website Upload Error",
                                        f"File: {Path(path).name}",
                                        f"Reason: {reason or 'Unknown website upload failure'}",
                                    ]
                                ),
                            )
                    except Exception as site_exc:
                        notify(
                            bot_token,
                            notify_chat_id,
                            "\n".join(
                                [
                                    "[ERROR] BNS Website Upload Exception",
                                    f"File: {Path(path).name}",
                                    f"Error: {site_exc}",
                                ]
                            ),
                        )

                # Re-upload to target channel using the date-named file
                print(
                    f"Uploading to target channel {args.target} with filename {Path(path).name}..."
                )
                await client.send_file(tgt_ent, path, caption=caption)
                print(f"Uploaded message {msg.id} -> target")

                # update state
                state["last_id"] = msg.id
                save_state(state_file, state)

            except errors.rpcerrorlist.PeerIdInvalidError as e:
                print(f"Permission error when sending to target: {e}")
                return
            except Exception as e:
                print(f"Error processing message {msg.id}: {e}")
            finally:
                # best-effort cleanup
                with contextlib.suppress(Exception):
                    if "path" in locals():
                        p = Path(path)
                        if p.exists():
                            p.unlink()
                with contextlib.suppress(Exception):
                    import shutil

                    shutil.rmtree(tempd, ignore_errors=True)

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
