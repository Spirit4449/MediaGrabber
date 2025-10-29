# downloader.py
import asyncio, os, re, json, argparse, contextlib
from telethon import TelegramClient, types, functions

API_ID = int(os.environ.get("TELEGRAM_API_ID", "0"))
API_HASH = os.environ.get("TELEGRAM_API_HASH", "")

if not API_ID or not API_HASH:
    raise SystemExit("TELEGRAM_API_ID / TELEGRAM_API_HASH env vars are required")

client = TelegramClient('media_grabber_session', API_ID, API_HASH,
                        timeout=60, request_retries=5, connection_retries=5)

def emit(event_type, **data):
    print(json.dumps({"type": event_type, **data}, ensure_ascii=False), flush=True)

def parse_link(link: str):
    link = link.strip()
    m = re.fullmatch(r'https?://t\.me/c/(\d+)/(\d+)', link)
    if m:
        chat_id = int(m.group(1)); msg_id = int(m.group(2))
        return ("peer_id", int(f"-100{chat_id}"), msg_id, True)  # is_private=True
    m = re.fullmatch(r'https?://t\.me/([A-Za-z0-9_]+)/(\d+)', link)
    if m:
        return ("username", m.group(1), int(m.group(2)), False)
    raise ValueError("Unsupported Telegram link format")

def parse_invite(inv: str) -> str:
    """Return invite hash from t.me/+xxxx or t.me/joinchat/xxxx or raw hash."""
    inv = inv.strip()
    m = re.search(r't\.me/(?:\+|joinchat/)([A-Za-z0-9_-]+)', inv)
    if m: return m.group(1)
    # If user pasted raw hash, accept it
    return inv

async def ensure_joined_with_invite(invite: str):
    # ImportChatInviteRequest expects the hash without prefix
    inv_hash = parse_invite(invite)
    res = await client(functions.messages.ImportChatInviteRequest(hash=inv_hash))
    return res

async def resolve_entity(kind, ref):
    await client.get_dialogs(limit=None)
    try:
        return await client.get_input_entity(ref)
    except Exception:
        if kind == "peer_id" and isinstance(ref, int):
            return await client.get_input_entity(types.PeerChannel(abs(ref)))
        raise

def human(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def compute_stall_timeout(expected_bytes: int | None):
    if not expected_bytes: return 90
    size_mb = expected_bytes / (1024 * 1024)
    return max(60, min(600, int(0.6 * size_mb)))

async def download_with_progress(msg, folder, progress_event):
    last = 0
    total = getattr(getattr(msg, "document", None), "size", None) or 0

    def cb(downloaded, _total):
        nonlocal last
        progress_event.set()
        if downloaded - last >= 512*1024 or downloaded == _total:
            last = downloaded
            pct = (downloaded / _total * 100) if _total else None
            emit("progress",
                 downloaded=downloaded, total=_total,
                 pct=(round(pct,1) if pct is not None else None),
                 text=f"⬇️ {human(downloaded)} / {human(_total) if _total else '??'}")

    os.makedirs(folder, exist_ok=True)
    path = await msg.download_media(file=folder, progress_callback=cb)
    return path

async def watchdog(task, progress_event, stall):
    try:
        while not task.done():
            progress_event.clear()
            try:
                await asyncio.wait_for(progress_event.wait(), timeout=stall)
            except asyncio.TimeoutError:
                task.cancel()
                emit("error", code="stall", text=f"⏱️ No progress for {stall}s")
                return
    except asyncio.CancelledError:
        return

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", required=True)
    ap.add_argument("--outdir", default=os.path.join(os.getcwd(), "downloads"))
    ap.add_argument("--invite", help="Optional invite link/hash to join before downloading")
    ap.add_argument("--preflight", action="store_true", help="Only check access and metadata; do not download")
    args = ap.parse_args()

    try:
        kind, ref, msg_id, is_private = parse_link(args.link)
    except Exception as e:
        emit("error", code="bad_link", text=f"❌ {e}")
        return

    try:
        await client.start()
        emit("status", text="✅ Logged in")

        # If entity cannot be resolved and it’s private, we may need invite
        need_invite = False
        entity = None
        try:
            entity = await resolve_entity(kind, ref)
        except Exception:
            need_invite = True

        if need_invite and args.invite:
            try:
                emit("status", text="🔗 Using invite to join…")
                await ensure_joined_with_invite(args.invite)
                entity = await resolve_entity(kind, ref)
                need_invite = False
                emit("status", text="✅ Joined via invite")
            except Exception as e:
                emit("error", code="invite_failed", text=f"❌ Invite join failed: {e}")
                return

        if args.preflight:
            if need_invite:
                emit("need_invite", text="🔐 Not a member; invite required", private=is_private)
                return
            # fetch the message to detect media and size
            msg = await client.get_messages(entity, ids=msg_id)
            if not msg:
                emit("error", code="not_found", text="⚠️ Message not found or no access")
                return
            expected = getattr(getattr(msg, "document", None), "size", None)
            emit("ok", text="✅ Access OK", expected=expected, has_media=bool(msg.media))
            return

        # Normal download flow
        if need_invite:
            emit("need_invite", text="🔐 Not a member; invite required", private=is_private)
            return

        msg = await client.get_messages(entity, ids=msg_id)
        if not msg:
            emit("error", code="not_found", text="⚠️ Message not found or no access"); return
        if not msg.media:
            emit("error", code="no_media", text="⚠️ No media in this post"); return

        chat_name = getattr(msg.chat, "username", None) or getattr(msg.chat, "title", None) or str(msg.chat.id)
        folder = os.path.join(args.outdir, chat_name)

        expected = getattr(getattr(msg, "document", None), "size", None)
        if expected:
            emit("status", text=f"📦 Expected size: {human(expected)}", expected=expected)

        stall = compute_stall_timeout(expected)
        emit("status", text=f"🛡️ No-progress watchdog: {stall}s", stall=stall)

        progress_event = asyncio.Event()
        dl_task = asyncio.create_task(download_with_progress(msg, folder, progress_event))
        wd_task = asyncio.create_task(watchdog(dl_task, progress_event, stall))

        try:
            path = await dl_task
        except asyncio.CancelledError:
            return
        finally:
            wd_task.cancel()
            with contextlib.suppress(Exception):
                await wd_task

        size = os.path.getsize(path) if os.path.exists(path) else None
        emit("done", text=f"✅ Done: {path}", path=path, size=size)
    except Exception as e:
        emit("error", code="exception", text=f"❌ {e.__class__.__name__}: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
