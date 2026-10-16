#!/usr/bin/env python3
"""
moments_sync.py

Send the "Moments with Mahant Swami" Telegram posts to the Brahma naa Sange site,
then wait until today's post has been turned into a prasang there.

The channel holds one short text post a day, so every run reads all of it: the same
command is the initial backfill, the daily import, and the catch-up for posts that
were edited or deleted later. The site stores posts by message id and ignores
unchanged ones, so repeats and retries are harmless. Images are not downloaded.

Requires TELEGRAM_API_ID, TELEGRAM_API_HASH and TELEGRAM_SESSION, plus the BNS admin
login used by daily_bns_sync.py (BNS_SITE_URL, BNS_ADMIN_USERNAME, BNS_ADMIN_PASSWORD).
"""
from __future__ import annotations

import argparse
import asyncio
import http.cookiejar
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from telethon import TelegramClient  # type: ignore

from daily_bns_sync import _resolve_bns_admin_credentials, _truncate, notify, resolve_entity_safely
from download_session import load_download_session
from reliability import exclusive_lock, load_state, save_state

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_CHANNEL = int(os.environ.get("MOMENTS_CHANNEL_ID", "2074676830"))
DEFAULT_STATE_FILE = Path(".state") / "moments_state.json"
FINISHED = {"done", "skipped"}


def post_payload(msg) -> dict | None:
    """The fields BNS stores for one message, or None when it carries no text."""
    text = getattr(msg, "message", None) or ""
    if not text.strip() or getattr(msg, "date", None) is None:
        return None
    edited = getattr(msg, "edit_date", None)
    grouped = getattr(msg, "grouped_id", None)
    return {
        "messageId": msg.id,
        "postedAt": msg.date.isoformat(),
        "editedAt": edited.isoformat() if edited else "",
        "text": text,
        "groupedId": str(grouped) if grouped else "",
        "hasPhoto": bool(getattr(msg, "photo", None)),
    }


def local_date(iso: str, zone: ZoneInfo) -> str:
    return datetime.fromisoformat(iso).astimezone(zone).date().isoformat()


class BnsSite:
    """Signed-in JSON calls to the BNS site. Uses the same admin login as the audio upload."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def _call(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json", "Accept": "application/json"})
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

    def import_posts(self, channel_id: int, username: str, posts: list[dict], complete: bool) -> dict:
        return self._call("POST", "/api/moments/import",
                          {"channelId": channel_id, "channelUsername": username, "posts": posts, "complete": complete})

    def status(self, message_ids: list[int]) -> list[dict]:
        return self._call("GET", "/api/moments/status?ids=" + ",".join(str(i) for i in message_ids)) or []


def describe(status: dict) -> str:
    if status.get("status") == "skipped":
        return f"#{status['messageId']} skipped: {status.get('skipReason') or 'not a moment'}"
    line = f"#{status['messageId']} \"{status.get('title') or 'Untitled'}\""
    if status.get("link") == "linked":
        line += f"\nAlso told in {status.get('linkedRecording') or 'a recording'} (\"{status.get('linkedTitle')}\"); retold as one."
    elif status.get("link") == "candidate":
        line += (f"\nPossible match with \"{status.get('linkedTitle')}\" in {status.get('linkedRecording') or 'a recording'}."
                 "\nReview: node scripts/moments/links.mjs --show " + str(status["messageId"]))
    return line


def channel_ref(value: str) -> int | str:
    """A numeric id (Bot API "-100..." ids are accepted), or a @username / t.me link."""
    value = value.strip()
    if value.lstrip("-").isdigit():
        number = abs(int(value))
        return number - 1_000_000_000_000 if number > 1_000_000_000_000 else number
    return value.rsplit("/", 1)[-1].lstrip("@")


async def visible_channels(client) -> list[tuple[int, str]]:
    """Channels and groups this Telegram account has joined, as (id, title)."""
    return [(dialog.entity.id, dialog.name or "") for dialog in await client.get_dialogs(limit=None)
            if getattr(dialog, "is_channel", False) or getattr(dialog, "is_group", False)]


async def resolve_channel(client, channel: int | str):
    try:
        return await resolve_entity_safely(client, channel)
    except ValueError:
        # Telegram only resolves a channel id for accounts that have joined it.
        seen = await visible_channels(client)
        likely = [item for item in seen if any(word in item[1].lower() for word in ("moment", "mahant", "swami"))]
        listed = "\n".join(f"  {id}  {title}" for id, title in (likely or seen)[:40]) or "  (none)"
        raise RuntimeError(
            f"This Telegram account cannot see channel {channel}. Join the channel with this account "
            f"(or check the id), then run again. Pass --channel with an id below, a @username or a t.me link.\n"
            f"{'Channels that look related' if likely else 'Channels this account can see'}:\n{listed}") from None


async def read_channel(client, channel: int | str, limit: int | None) -> tuple[int, str, list[dict]]:
    entity = await resolve_channel(client, channel)
    full = await client.get_entity(entity)
    posts = []
    async for msg in client.iter_messages(entity, limit=limit):
        payload = post_payload(msg)
        if payload:
            posts.append(payload)
    posts.sort(key=lambda post: post["messageId"])
    return full.id, getattr(full, "username", "") or "", posts


async def run_sync(args) -> int:
    api_id = int(os.environ.get("TELEGRAM_API_ID", "0"))
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    session = args.session or os.environ.get("TELEGRAM_SESSION", "media_grabber_session.session")
    bot_token = os.environ.get("BOT_TOKEN", "")
    notify_chat_id = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "")
    zone = ZoneInfo(os.environ.get("BNS_TIMEZONE", "America/New_York"))
    if not api_id or not api_hash:
        raise RuntimeError("TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables are required")

    state_file = Path(args.state_file)
    state = load_state(state_file)
    today = datetime.now(zone).date().isoformat()
    if args.stop_after_today and state.get("last_delivery_date", "") >= today:
        print("Today's moment is already in the archive.")
        return 0

    client = TelegramClient(load_download_session(session), api_id, api_hash,
                            timeout=60, request_retries=5, connection_retries=5)
    try:
        await client.connect()
        if not await client.is_user_authorized():
            raise RuntimeError("Telegram session is not authorized; run downloader.py --login on this host")
        channel_id, username, posts = await read_channel(client, args.channel, args.limit or None)
    finally:
        await client.disconnect()
    print(f"Read {len(posts)} posts with text from channel {channel_id}"
          + (f" (@{username})" if username else ""))

    if args.dry_run:
        out = Path(args.out) if args.out else None
        if out:
            out.write_text("".join(json.dumps(post, ensure_ascii=False) + "\n" for post in posts), encoding="utf-8")
            print(f"Wrote {out}")
        if posts:
            print(f"Oldest {posts[0]['postedAt'][:10]}, newest {posts[-1]['postedAt'][:10]}; "
                  f"{sum(p['hasPhoto'] for p in posts)} with a photo, {sum(bool(p['editedAt']) for p in posts)} edited, "
                  f"{sum(bool(p['groupedId']) for p in posts)} in albums.")
        return 0

    site_url = os.environ.get("BNS_SITE_URL", "https://bns.classchats.net")
    username_bns, password = _resolve_bns_admin_credentials()
    if not username_bns or not password:
        raise RuntimeError("Set BNS_ADMIN_USERNAME and BNS_ADMIN_PASSWORD to import moments")
    site = BnsSite(site_url)
    site.login(username_bns, password)
    # Only a full read may mark missing posts as deleted.
    result = site.import_posts(channel_id, username, posts, complete=not args.limit)
    print(f"BNS import: {result.get('inserted', 0)} new, {result.get('updated', 0)} changed, {result.get('unchanged', 0)} unchanged")

    todays = [post["messageId"] for post in posts if local_date(post["postedAt"], zone) == today]
    if not todays:
        print(f"No post dated {today} yet; will check again.")
        return 3

    deadline = time.monotonic() + args.wait
    while True:
        statuses = site.status(todays)
        if statuses and all(item.get("status") in FINISHED for item in statuses):
            break
        if time.monotonic() >= deadline:
            waiting = "; ".join(f"#{s['messageId']} {s.get('status')}" + (f" ({_truncate(s['error'], 200)})" if s.get("error") else "")
                                for s in statuses if s.get("status") not in FINISHED) or "no status yet"
            raise RuntimeError(f"Today's moment is not processed yet: {waiting}. The site keeps retrying; this check runs again.")
        time.sleep(args.poll)

    state["last_delivery_date"] = max(state.get("last_delivery_date", ""), today)
    state["last_success_at"] = time.time()
    save_state(state_file, state)
    notify(bot_token, notify_chat_id, "[OK] Moments with Mahant Swami\n" + "\n\n".join(describe(item) for item in statuses))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", type=channel_ref, default=DEFAULT_CHANNEL,
                    help="Moments channel: numeric id, @username or t.me link")
    ap.add_argument("--state-file", default=str(DEFAULT_STATE_FILE), help="Holds the date of the last processed post.")
    ap.add_argument("--session", default="", help="Override TELEGRAM_SESSION for this run.")
    ap.add_argument("--dry-run", action="store_true", help="Read the channel and report; send nothing to the site.")
    ap.add_argument("--out", default="", help="With --dry-run, write the posts as JSON lines to this file.")
    ap.add_argument("--limit", type=int, default=0, help="Only the newest N posts (pilot). Deletions are not detected.")
    ap.add_argument("--wait", type=int, default=int(os.environ.get("MOMENTS_WAIT_SECONDS", "600")),
                    help="Seconds to wait for today's post to be processed.")
    ap.add_argument("--poll", type=int, default=15, help=argparse.SUPPRESS)
    ap.add_argument("--stop-after-today", action="store_true",
                    help="Do nothing once today's post is processed (used by the scheduled worker).")
    args = ap.parse_args()
    with exclusive_lock(str(Path(args.state_file).resolve()) + ".lock"):
        return asyncio.run(run_sync(args))


if __name__ == "__main__":
    raise SystemExit(main())
