import asyncio, os, re, contextlib
from telethon import TelegramClient, types

# ---- Your API credentials ----
api_id = 27434653
api_hash = '41a4175c8c019f5b61b6484af9d02c64'

client = TelegramClient(
    'media_grabber_session',
    api_id,
    api_hash,
    timeout=60,
    request_retries=5,
    connection_retries=5
)

def parse_link(link: str):
    link = link.strip()
    # Private/supergroup: https://t.me/c/<chatId>/<msgId>
    m = re.fullmatch(r'https?://t\.me/c/(\d+)/(\d+)', link)
    if m:
        chat_id = int(m.group(1)); msg_id = int(m.group(2))
        peer_id = int(f"-100{chat_id}")
        return ("peer_id", peer_id, msg_id)
    # Public: https://t.me/<username>/<msgId>
    m = re.fullmatch(r'https?://t\.me/([A-Za-z0-9_]+)/(\d+)', link)
    if m:
        username = m.group(1); msg_id = int(m.group(2))
        return ("username", username, msg_id)
    raise ValueError("Unsupported Telegram link format.")

async def resolve_entity(kind, ref):
    await client.get_dialogs(limit=None)  # warm cache
    try:
        return await client.get_input_entity(ref)
    except Exception:
        if kind == "peer_id" and isinstance(ref, int):
            return await client.get_input_entity(types.PeerChannel(abs(ref)))
        raise

def human(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def compute_stall_timeout(expected_bytes: int | None):
    """
    Stall timeout (no-progress) scales with size.
    If unknown size: 90s. Otherwise 0.6s/MB clamped to [60s, 600s].
    """
    if not expected_bytes:
        return 90
    size_mb = expected_bytes / (1024 * 1024)
    return max(60, min(600, int(0.6 * size_mb)))

async def download_with_progress(msg, folder, progress_event):
    last_print = 0
    total = getattr(getattr(msg, "document", None), "size", None) or 0

    def callback(downloaded, _total):
        nonlocal last_print
        # mark progress for watchdog
        progress_event.set()
        # throttle prints
        if downloaded - last_print >= 512 * 1024 or downloaded == _total:
            last_print = downloaded
            if _total:
                pct = 100 * downloaded / _total
                print(f"\rDownloading... {human(downloaded)} / {human(_total)} ({pct:.1f}%)", end="", flush=True)

    os.makedirs(folder, exist_ok=True)
    path = await msg.download_media(file=folder, progress_callback=callback)
    print()  # newline after progress
    return path

async def no_progress_watchdog(download_task, progress_event, stall_timeout_s):
    """
    Cancels the download if there's no progress for stall_timeout_s.
    Resets the timer whenever progress_event is set.
    """
    try:
        while not download_task.done():
            progress_event.clear()
            try:
                await asyncio.wait_for(progress_event.wait(), timeout=stall_timeout_s)
            except asyncio.TimeoutError:
                download_task.cancel()
                return
    except asyncio.CancelledError:
        return

async def main(link: str):
    progress_event = asyncio.Event()
    try:
        kind, ref, msg_id = parse_link(link)
        await client.start()
        print("Logged in successfully.")

        entity = await resolve_entity(kind, ref)
        message = await client.get_messages(entity, ids=msg_id)
        if not message:
            print("⚠️ Message not found or no access.")
            return
        if not message.media:
            print("⚠️ No media found in this post.")
            return

        base = os.path.join(os.getcwd(), "downloads")
        chat_name = getattr(message.chat, "username", None) or getattr(message.chat, "title", None) or str(message.chat.id)
        folder = os.path.join(base, chat_name)

        expected = getattr(getattr(message, "document", None), "size", None)
        if expected:
            print(f"Expected size: {human(expected)}")

        stall_timeout_s = compute_stall_timeout(expected)
        print(f"No-progress watchdog: {stall_timeout_s}s")

        # Start download + watchdog
        download_task = asyncio.create_task(download_with_progress(message, folder, progress_event))
        watchdog_task = asyncio.create_task(no_progress_watchdog(download_task, progress_event, stall_timeout_s))

        try:
            file_path = await download_task  # no overall timeout
            print(f"✅ Done: {file_path}")
        except asyncio.CancelledError:
            print(f"\n❌ Error: No progress for {stall_timeout_s}s, download cancelled.")
        finally:
            watchdog_task.cancel()
            with contextlib.suppress(Exception):
                await watchdog_task

    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    link = input("Enter Telegram post link: ").strip()
    asyncio.run(main(link))
