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
import tempfile
import time
import urllib.parse
import urllib.request
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
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    options = webdriver.ChromeOptions()
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,1400")

    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(120)
        driver.get(site_url)
        wait = WebDriverWait(driver, 30)

        admin_btn = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[normalize-space()='Are you an admin?']")
            )
        )
        driver.execute_script("arguments[0].click();", admin_btn)

        user_input = wait.until(
            EC.visibility_of_element_located((By.XPATH, "//input[@placeholder='Admin username']"))
        )
        pass_input = wait.until(
            EC.visibility_of_element_located((By.XPATH, "//input[@placeholder='Admin password']"))
        )
        user_input.clear()
        user_input.send_keys(admin_username)
        pass_input.clear()
        pass_input.send_keys(admin_password)

        verify_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='Verify']"))
        )
        driver.execute_script("arguments[0].click();", verify_btn)

        wait.until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(normalize-space(),'Admin:')]"))
        )

        file_input = wait.until(EC.presence_of_element_located((By.ID, "audio-upload")))
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
        file_name_lit = _xpath_literal(file_name)
        item_xpath = (
            f"//h4[normalize-space()={file_name_lit}]"
            "/ancestor::div[contains(@class,'bg-white')][1]"
        )
        wait.until(EC.presence_of_element_located((By.XPATH, item_xpath)))

        deadline = time.time() + max(60, upload_timeout_seconds)
        last_card_text = ""
        last_change_at = time.time()
        while time.time() < deadline:
            cards = driver.find_elements(By.XPATH, item_xpath)
            browser_excerpt = _browser_error_excerpt(driver)
            if cards:
                card = cards[0]
                card_text = card.text.strip()
                if card_text != last_card_text:
                    last_card_text = card_text
                    last_change_at = time.time()

                compact_text = "".join(card_text.upper().split())
                if "Successfully Archived" in card_text or "Archived (rate limits detected/recovered)" in card_text:
                    title, summary = _extract_recent_upload_summary(driver)
                    return {
                        "status": "uploaded",
                        "reason": "Completed",
                        "title": title,
                        "summary": summary,
                    }
                if (
                    "ERROR" in compact_text
                    or "PROCESSINGFAILED" in compact_text
                    or "RESOURCE_EXHAUSTED" in compact_text
                ):
                    err_text = _extract_card_error(card, browser_excerpt)
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

        browser_excerpt = _browser_error_excerpt(driver)
        reason = f"Timed out waiting for upload completion after {upload_timeout_seconds}s"
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

    can_site_upload = (
        (not skip_site_upload)
        and bool(admin_username)
        and bool(admin_password)
    )
    if not can_site_upload and not skip_site_upload:
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

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

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
                # ensure unique target name inside outdir
                candidate = outdir / base_name
                if not candidate.exists():
                    final_path = candidate
                else:
                    r = candidate.with_suffix("")
                    e = candidate.suffix
                    i = 1
                    while True:
                        cand = outdir / f"{r.name} ({i}){e}"
                        if not cand.exists():
                            final_path = cand
                            break
                        i += 1

                # move downloaded file to final_path
                try:
                    Path(path).rename(final_path)
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
