# downloader.py
import asyncio, os, re, json, argparse, contextlib
from telethon import TelegramClient, types, functions, utils  # <-- utils helps with extensions

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
        return ("peer_id", int(f"-100{chat_id}"), msg_id, True)
    m = re.fullmatch(r'https?://t\.me/([A-Za-z0-9_]+)/(\d+)', link)
    if m:
        return ("username", m.group(1), int(m.group(2)), False)
    raise ValueError("Unsupported Telegram link format")

def parse_invite(inv: str) -> str:
    inv = inv.strip()
    m = re.search(r't\.me/(?:\+|joinchat/)([A-Za-z0-9_-]+)', inv)
    if m: return m.group(1)
    return inv

async def ensure_joined_with_invite(invite: str):
    inv_hash = parse_invite(invite)
    return await client(functions.messages.ImportChatInviteRequest(hash=inv_hash))

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

# ---------- filename helpers ----------
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
    "application/pdf": ".pdf",
    "application/zip": ".zip",
    "application/x-rar-compressed": ".rar",
}

def _sanitize_filename(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in (" ", ".", "-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip().strip(".")
    return out or "file"

def _ext_from_media(msg) -> str:
    """
    Prefer Telethon's utils.get_extension for documents/photos/voice/sticker.
    Fallback to known MIME mappings, then .bin.
    """
    # Document (includes videos/voice/stickers sent as "file")
    if getattr(msg, "document", None):
        ext = utils.get_extension(msg.document)  # e.g., ".mp4", ".pdf", ".webp"
        if ext:
            return ext
        mime = getattr(msg.document, "mime_type", None)
        return MIME_EXT_FALLBACK.get(mime, ".bin")
    # Photo
    if getattr(msg, "photo", None):
        return ".jpg"  # Telegram photos are JPEG on download
    # Anything else (rare)
    f = getattr(msg, "file", None)
    if f and getattr(f, "ext", None):
        return f.ext
    return ".bin"

def _name_from_attr(document):
    """
    If the sender provided a filename (DocumentAttributeFilename), use it.
    """
    if not document:
        return None
    for attr in getattr(document, "attributes", []) or []:
        if isinstance(attr, types.DocumentAttributeFilename) and attr.file_name:
            return attr.file_name
    return None

def _derive_target_path(msg, folder: str) -> str:
    # Prefer sender's filename if present
    orig = _name_from_attr(getattr(msg, "document", None))
    if not orig and getattr(msg, "file", None):
        # Sometimes Telethon exposes a name here
        orig = getattr(msg.file, "name", None)

    if orig:
        base = _sanitize_filename(orig)
        root, ext = os.path.splitext(base)
        if not ext:
            base = base + _ext_from_media(msg)
    else:
        # Build from chat tag and message id
        chat_tag = getattr(msg.chat, "username", None) or getattr(msg.chat, "title", None) or "file"
        chat_tag = _sanitize_filename(str(chat_tag))
        base = f"{chat_tag}_{msg.id}{_ext_from_media(msg)}"

    # Ensure uniqueness
    os.makedirs(folder, exist_ok=True)
    candidate = os.path.join(folder, base)
    if not os.path.exists(candidate):
        return candidate
    r, e = os.path.splitext(candidate)
    i = 1
    while True:
        cand = f"{r} ({i}){e}"
        if not os.path.exists(cand):
            return cand
        i += 1

# ---------- download & watchdog ----------
async def download_with_progress(msg, folder, progress_event):
    last = 0
    total = getattr(getattr(msg, "document", None), "size", None) or 0
    target_path = _derive_target_path(msg, folder)

    def cb(downloaded, _total):
        nonlocal last
        progress_event.set()
        if downloaded - last >= 512 * 1024 or downloaded == _total:
            last = downloaded
            pct = (downloaded / _total * 100) if _total else None
            emit("progress",
                 downloaded=downloaded, total=_total,
                 pct=(round(pct, 1) if pct is not None else None))

    # Pass a FULL file path with extension so Telethon won't create document.dat
    path = await msg.download_media(file=target_path, progress_callback=cb)
    return path

async def watchdog(task, progress_event, stall):
    try:
        while not task.done():
            progress_event.clear()
            try:
                await asyncio.wait_for(progress_event.wait(), timeout=stall)
            except asyncio.TimeoutError:
                task.cancel()
                emit("error", code="stall", text=f"No progress for {stall}s")
                return
    except asyncio.CancelledError:
        return

# ---------- main ----------
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
        emit("error", code="bad_link", text=f"{e}")
        return

    try:
        await client.start()
        emit("status", text="Logged in")

        need_invite = False
        try:
            entity = await resolve_entity(kind, ref)
        except Exception:
            need_invite = True
            entity = None

        if need_invite and args.invite:
            try:
                emit("status", text="Joining via invite…")
                await ensure_joined_with_invite(args.invite)
                entity = await resolve_entity(kind, ref)
                need_invite = False
                emit("status", text="Joined")
            except Exception as e:
                emit("error", code="invite_failed", text=f"Invite join failed: {e}")
                return

        if args.preflight:
            if need_invite:
                emit("need_invite", text="Invite required", private=is_private)
                return
            msg = await client.get_messages(entity, ids=msg_id)
            if not msg:
                emit("error", code="not_found", text="Message not found or no access")
                return
            expected = getattr(getattr(msg, "document", None), "size", None)
            emit("ok", expected=expected, has_media=bool(msg.media))
            return

        if need_invite:
            emit("need_invite", text="Invite required", private=is_private)
            return

        msg = await client.get_messages(entity, ids=msg_id)
        if not msg:
            emit("error", code="not_found", text="Message not found or no access"); return
        if not msg.media:
            emit("error", code="no_media", text="No media in this post"); return

        chat_name = getattr(msg.chat, "username", None) or getattr(msg.chat, "title", None) or str(msg.chat.id)
        folder = os.path.join(args.outdir, _sanitize_filename(str(chat_name)))

        expected = getattr(getattr(msg, "document", None), "size", None)
        if expected:
            emit("status", text=f"Expected size: {expected}", expected=expected)

        stall = compute_stall_timeout(expected)
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
        emit("done", path=path, size=size)
    except Exception as e:
        emit("error", code="exception", text=f"{e.__class__.__name__}: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
