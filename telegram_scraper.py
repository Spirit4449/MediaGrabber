#!/usr/bin/env python3
"""Telegram image scraper with Gemini-based categorization."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError
from telethon import TelegramClient, errors
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.custom import Message


class ConfigError(Exception):
    """Raised when the configuration file is invalid."""


class GeminiQuotaExceeded(Exception):
    """Raised when Gemini rate limits are exhausted."""


class GeminiUnavailable(Exception):
    """Raised when Gemini cannot be initialised."""


@dataclass
class ImageRecord:
    """Represents a downloaded Telegram image."""

    path: Path
    channel: str
    message_id: int
    message_date: datetime


class JSONStore:
    """Simple async-safe JSON persistence helper."""

    def __init__(self, path: Path, default: Any, logger: logging.Logger) -> None:
        self.path = path
        self.default = default
        self.logger = logger
        self._lock = asyncio.Lock()
        self.data = self._load()

    def _load(self) -> Any:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except json.JSONDecodeError as err:
                self.logger.warning("Failed to parse %s (%s); using default state", self.path, err)
        return json.loads(json.dumps(self.default))

    async def mutate(self, mutator: Callable[[Any], None]) -> None:
        async with self._lock:
            mutator(self.data)
            atomic_write(self.path, self.data)

    async def write(self) -> None:
        async with self._lock:
            atomic_write(self.path, self.data)


class RunStats:
    """Tracks run-time counters with async safety."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._data: Dict[str, int] = {
            "scraped": 0,
            "categorized": 0,
            "duplicates": 0,
            "pending": 0,
            "failed": 0,
        }

    async def increment(self, key: str, amount: int = 1) -> None:
        async with self._lock:
            self._data[key] = self._data.get(key, 0) + amount

    def snapshot(self) -> Dict[str, int]:
        return dict(self._data)


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
RESULTS_PATH = BASE_DIR / "results.json"
PROGRESS_PATH = BASE_DIR / "progress.json"
PENDING_PATH = BASE_DIR / "pending_images.json"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "telegram_agent.log"
TEMP_DIR = BASE_DIR / "data" / "tmp"
PENDING_DIR = BASE_DIR / "data" / "pending"
DEFAULT_PROMPT = (
    "You are a creative design assistant. Generate 10 descriptive tags for this image "
    "and rate whether it is a 'good design inspiration' or 'not good'. Output tags as a "
    "JSON array and include a boolean field 'is_good'."
)
DEFAULT_MODEL_NAME = "gemini-1.5-flash"
DEFAULT_DAILY_TIME = (2, 0)
RESOURCE_EXHAUSTED_ERROR = getattr(google_exceptions, "ResourceExhausted", Exception)
TOO_MANY_REQUESTS_ERROR = getattr(google_exceptions, "TooManyRequests", RESOURCE_EXHAUSTED_ERROR)


def ensure_directories() -> None:
    for directory in (LOG_DIR, TEMP_DIR, PENDING_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file at {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = ["api_id", "api_hash", "channels", "start_date"]
    for key in required:
        if key not in config or not config[key]:
            raise ConfigError(f"Config field '{key}' is required")
    channels = [str(ch).strip() for ch in config.get("channels", []) if str(ch).strip()]
    if not channels:
        raise ConfigError("'channels' must contain at least one channel username")
    config["channels"] = channels
    try:
        config["api_id"] = int(config["api_id"])
    except (TypeError, ValueError) as err:
        raise ConfigError("'api_id' must be an integer") from err
    config["start_date"] = str(config["start_date"])
    config.setdefault("gemini_model", DEFAULT_MODEL_NAME)
    config.setdefault("daily_run_time", "02:00")
    config.setdefault("max_concurrency", 3)
    return config


def parse_start_date(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as err:
        raise ConfigError("'start_date' must be ISO 8601 formatted") from err
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_daily_time(value: Optional[str], logger: logging.Logger) -> Tuple[int, int]:
    if not value:
        return DEFAULT_DAILY_TIME
    try:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("Expected HH:MM format")
        hour = max(0, min(23, int(parts[0])))
        minute = max(0, min(59, int(parts[1])))
        return hour, minute
    except (ValueError, TypeError) as err:
        logger.warning("Invalid daily_run_time '%s' (%s); defaulting to %02d:%02d", value, err, *DEFAULT_DAILY_TIME)
        return DEFAULT_DAILY_TIME


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("telegram_agent")
    if logger.handlers:
        return logger
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def apply_log_level(logger: logging.Logger, level: Optional[str]) -> None:
    if not level:
        return
    numeric_level = getattr(logging, str(level).upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)


def compute_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def message_has_image(message: Message) -> bool:
    if not message or not message.media:
        return False
    mime = getattr(message, "file", None)
    if not mime:
        return False
    mime_type = getattr(message.file, "mime_type", "")
    return bool(mime_type and mime_type.startswith("image/"))


async def scrape_channel(
    client: TelegramClient,
    channel: str,
    start_date: datetime,
    last_message_id: int,
    tmp_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[ImageRecord], int]:
    images: List[ImageRecord] = []
    new_last_id = last_message_id
    async for message in client.iter_messages(
        channel,
        min_id=last_message_id,
        min_date=start_date,
        reverse=True,
    ):
        if not isinstance(message, Message):
            continue
        message_date = ensure_utc_datetime(message.date)
        new_last_id = max(new_last_id, message.id or new_last_id)
        if message_date < start_date:
            continue
        if not message_has_image(message):
            continue
        download_target = tmp_dir / f"{channel.strip('@')}_{message.id}"
        try:
            path_str = await message.download_media(file=download_target)
        except FloodWaitError as wait_err:
            wait_seconds = max(wait_err.seconds, 1)
            logger.warning(
                "Flood wait for %s (%ss); backing off", channel, wait_seconds
            )
            await asyncio.sleep(wait_seconds)
            continue
        except RPCError as rpc_err:
            logger.error("Failed to download media from %s: %s", channel, rpc_err)
            continue
        if not path_str:
            continue
        media_path = Path(path_str)
        if not media_path.exists():
            continue
        images.append(
            ImageRecord(
                path=media_path,
                channel=channel,
                message_id=message.id,
                message_date=message_date,
            )
        )
    if images:
        logger.info("Downloaded %d image(s) from %s", len(images), channel)
    return images, new_last_id


def ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso_date_from_string(value: Optional[str]) -> str:
    if not value:
        return datetime.now(timezone.utc).date().isoformat()
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc).date().isoformat()
    return ensure_utc_datetime(parsed).date().isoformat()


def get_unique_path(folder: Path, filename: str) -> Path:
    candidate = folder / filename
    if not candidate.exists():
        return candidate
    stem, suffix = os.path.splitext(filename)
    counter = 1
    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


async def process_images(
    records: List[ImageRecord],
    model: Optional[genai.GenerativeModel],
    prompt: str,
    results_store: JSONStore,
    progress_store: JSONStore,
    pending_store: JSONStore,
    processed_hashes: Set[str],
    semaphore: asyncio.Semaphore,
    stats: RunStats,
    logger: logging.Logger,
) -> None:
    if not records:
        return
    await stats.increment("scraped", len(records))
    tasks = [
        asyncio.create_task(
            process_image(
                record,
                model,
                prompt,
                results_store,
                progress_store,
                pending_store,
                processed_hashes,
                semaphore,
                stats,
                logger,
            )
        )
        for record in records
    ]
    await asyncio.gather(*tasks)


async def process_image(
    record: ImageRecord,
    model: Optional[genai.GenerativeModel],
    prompt: str,
    results_store: JSONStore,
    progress_store: JSONStore,
    pending_store: JSONStore,
    processed_hashes: Set[str],
    semaphore: asyncio.Semaphore,
    stats: RunStats,
    logger: logging.Logger,
) -> None:
    await semaphore.acquire()
    try:
        try:
            content_hash = await asyncio.to_thread(compute_md5, record.path)
        except FileNotFoundError:
            await stats.increment("failed")
            return
        if content_hash in processed_hashes:
            await stats.increment("duplicates")
            logger.info("Duplicate image skipped: %s", record.path.name)
            record.path.unlink(missing_ok=True)
            return
        if model is None:
            await move_to_pending(
                record,
                content_hash,
                pending_store,
                processed_hashes,
                progress_store,
                stats,
                logger,
                reason="model_unavailable",
            )
            return
        try:
            tags, is_good = await categorize_image(model, prompt, record.path, logger)
        except GeminiQuotaExceeded:
            logger.warning("Gemini quota reached; deferring %s", record.path.name)
            await move_to_pending(
                record,
                content_hash,
                pending_store,
                processed_hashes,
                progress_store,
                stats,
                logger,
                reason="rate_limit",
            )
            return
        except GeminiUnavailable as err:
            logger.error("Gemini unavailable: %s", err)
            await move_to_pending(
                record,
                content_hash,
                pending_store,
                processed_hashes,
                progress_store,
                stats,
                logger,
                reason="unavailable",
            )
            return
        except Exception as exc:
            await stats.increment("failed")
            logger.exception("Categorization failed for %s: %s", record.path.name, exc)
            record.path.unlink(missing_ok=True)
            return
        result_entry = build_result_entry(record, content_hash, tags, is_good)
        await results_store.mutate(lambda data: data.append(result_entry))
        await register_hash(content_hash, processed_hashes, progress_store)
        await stats.increment("categorized")
        record.path.unlink(missing_ok=True)
        logger.info(
            "Categorized %s from %s -> %s",
            record.path.name,
            record.channel,
            "good" if is_good else "not good",
        )
    finally:
        semaphore.release()


async def move_to_pending(
    record: ImageRecord,
    content_hash: str,
    pending_store: JSONStore,
    processed_hashes: Set[str],
    progress_store: JSONStore,
    stats: RunStats,
    logger: logging.Logger,
    reason: str,
) -> None:
    destination = get_unique_path(PENDING_DIR, record.path.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(record.path), destination)
    pending_entry = {
        "hash": content_hash,
        "file": str(destination),
        "original_filename": record.path.name,
        "source_channel": record.channel,
        "message_id": record.message_id,
        "message_date": record.message_date.isoformat(),
        "added_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    await pending_store.mutate(lambda data: data.append(pending_entry))
    await register_hash(content_hash, processed_hashes, progress_store)
    await stats.increment("pending")
    logger.info("Deferred %s for later categorization", record.path.name)


async def register_hash(content_hash: str, processed_hashes: Set[str], progress_store: JSONStore) -> None:
    if content_hash in processed_hashes:
        return
    processed_hashes.add(content_hash)
    await progress_store.mutate(
        lambda data: _append_unique_hash(data, content_hash)
    )


def _append_unique_hash(data: Dict[str, Any], content_hash: str) -> None:
    hashes = data.setdefault("processed_hashes", [])
    if content_hash not in hashes:
        hashes.append(content_hash)


def build_result_entry(
    record: ImageRecord,
    content_hash: str,
    tags: List[str],
    is_good: bool,
) -> Dict[str, Any]:
    return {
        "filename": record.path.name,
        "hash": content_hash,
        "tags": tags,
        "is_good": bool(is_good),
        "source_channel": record.channel,
        "date": record.message_date.date().isoformat(),
        "message_id": record.message_id,
        "message_timestamp": record.message_date.isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def categorize_image(
    model: genai.GenerativeModel,
    prompt: str,
    image_path: Path,
    logger: logging.Logger,
    max_retries: int = 5,
) -> Tuple[List[str], bool]:
    attempt = 0
    backoff = 5
    while True:
        attempt += 1
        try:
            return await asyncio.to_thread(run_gemini_generation, model, prompt, image_path)
        except GeminiQuotaExceeded:
            raise
        except GeminiUnavailable:
            raise
    except (RESOURCE_EXHAUSTED_ERROR, TOO_MANY_REQUESTS_ERROR) as limits:
            if attempt >= max_retries:
                raise GeminiQuotaExceeded(str(limits))
            logger.warning("Gemini rate limited (%s); retrying in %ss", limits, backoff)
            await asyncio.sleep(backoff)
            backoff *= 2
        except google_exceptions.GoogleAPIError as transient:
            if attempt >= max_retries:
                raise GeminiUnavailable(str(transient))
            logger.warning("Gemini transient error (%s); retrying in %ss", transient, backoff)
            await asyncio.sleep(backoff)
            backoff *= 2
        except Exception as exc:
            if attempt >= max_retries:
                raise GeminiUnavailable(str(exc))
            logger.warning("Gemini unexpected error (%s); retrying in %ss", exc, backoff)
            await asyncio.sleep(backoff)
            backoff *= 2


def run_gemini_generation(
    model: genai.GenerativeModel,
    prompt: str,
    image_path: Path,
) -> Tuple[List[str], bool]:
    try:
        with Image.open(image_path) as image:
            image.load()
            image_for_model = image.copy()
    except UnidentifiedImageError as err:
        raise GeminiUnavailable(f"Invalid image {image_path.name}: {err}") from err
    response = model.generate_content([prompt, image_for_model])
    if hasattr(response, "prompt_feedback") and getattr(response.prompt_feedback, "block_reason", None):
        raise GeminiUnavailable(f"Prompt blocked: {response.prompt_feedback.block_reason}")
    text = (response.text or "").strip()
    if not text and getattr(response, "candidates", None):
        fragments: List[str] = []
        for candidate in response.candidates:
            parts = getattr(candidate.content, "parts", [])
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    fragments.append(part_text)
        text = "\n".join(fragments)
    if not text:
        raise GeminiUnavailable("Gemini returned empty response")
    tags, is_good = parse_gemini_output(text)
    return tags, is_good


def parse_gemini_output(raw_text: str) -> Tuple[List[str], bool]:
    cleaned = strip_code_fences(raw_text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as err:
        raise GeminiUnavailable(f"Gemini response not JSON: {err}") from err
    if isinstance(payload, dict):
        tags = payload.get("tags")
        is_good = payload.get("is_good")
    else:
        raise GeminiUnavailable("Gemini response must be a JSON object with 'tags' and 'is_good'")
    if not isinstance(tags, list):
        raise GeminiUnavailable("Gemini response invalid: 'tags' must be a list")
    tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    tags = tags[:10]
    if not tags:
        raise GeminiUnavailable("Gemini response invalid: tags list empty")
    if isinstance(is_good, str):
        lowered = is_good.strip().lower()
        is_good_bool = lowered in {"true", "yes", "good", "good design inspiration", "1"}
    else:
        is_good_bool = bool(is_good)
    return tags, is_good_bool


def strip_code_fences(value: str) -> str:
    if value.startswith("```") and value.endswith("```"):
        parts = value.split("\n", 1)
        if len(parts) == 2:
            value = parts[1]
        if value.endswith("```"):
            value = value[: -3]
    return value.strip()


async def process_pending_items(
    model: Optional[genai.GenerativeModel],
    prompt: str,
    pending_store: JSONStore,
    results_store: JSONStore,
    processed_hashes: Set[str],
    progress_store: JSONStore,
    stats: RunStats,
    logger: logging.Logger,
) -> None:
    if not pending_store.data:
        return
    if model is None:
        logger.info("Gemini unavailable; skipping %d pending item(s)", len(pending_store.data))
        return
    pending_items = list(pending_store.data)
    for entry in pending_items:
        image_path = Path(entry["file"])
        if not image_path.exists():
            logger.warning("Pending file missing; discarding %s", image_path)
            await pending_store.mutate(
                lambda data, h=entry["hash"]: _remove_pending(data, h)
            )
            continue
        try:
            tags, is_good = await categorize_image(model, prompt, image_path, logger)
        except GeminiQuotaExceeded:
            logger.warning("Gemini quota hit while processing pending; stopping early")
            break
        except Exception as exc:
            logger.exception("Failed to process pending %s: %s", image_path.name, exc)
            continue
        result_entry = {
            "filename": entry.get("original_filename", image_path.name),
            "hash": entry["hash"],
            "tags": tags,
            "is_good": bool(is_good),
            "source_channel": entry.get("source_channel", "unknown"),
            "date": iso_date_from_string(entry.get("message_date")),
            "message_id": entry.get("message_id"),
            "message_timestamp": entry.get("message_date"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await results_store.mutate(lambda data, item=result_entry: data.append(item))
        await register_hash(entry["hash"], processed_hashes, progress_store)
        await pending_store.mutate(
            lambda data, h=entry["hash"]: _remove_pending(data, h)
        )
        image_path.unlink(missing_ok=True)
        await stats.increment("categorized")
        logger.info("Categorized pending %s", result_entry["filename"])


def _remove_pending(data: List[Dict[str, Any]], content_hash: str) -> None:
    data[:] = [item for item in data if item.get("hash") != content_hash]


def setup_gemini_client(api_key: Optional[str], model_name: str, logger: logging.Logger) -> Optional[genai.GenerativeModel]:
    if not api_key:
        logger.warning("Gemini API key not configured; categorization deferred")
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception as exc:
        logger.error("Failed to initialise Gemini client: %s", exc)
        return None


async def run_pipeline_job(logger: logging.Logger) -> None:
    logger.info("Starting Telegram scraping cycle")
    ensure_directories()
    try:
        config = load_config(CONFIG_PATH)
    except ConfigError as err:
        logger.error("Configuration error: %s", err)
        return
    apply_log_level(logger, config.get("log_level"))
    start_date = parse_start_date(config["start_date"])
    channels = config.get("channels", [])
    if not channels:
        logger.warning("No channels to process")
        return
    session_name = config.get("session_name", "telegram_scraper")
    session_path = Path(session_name)
    if not session_path.is_absolute():
        session_path = BASE_DIR / session_path
    api_id = config["api_id"]
    api_hash = str(config["api_hash"])
    phone_number = config.get("phone_number")
    model = setup_gemini_client(config.get("gemini_api_key"), config.get("gemini_model", DEFAULT_MODEL_NAME), logger)
    prompt = config.get("gemini_prompt") or DEFAULT_PROMPT
    results_store = JSONStore(RESULTS_PATH, [], logger)
    progress_store = JSONStore(PROGRESS_PATH, {"channels": {}, "processed_hashes": []}, logger)
    pending_store = JSONStore(PENDING_PATH, [], logger)
    processed_hashes = set(progress_store.data.get("processed_hashes", []))
    for entry in pending_store.data:
        processed_hashes.add(entry.get("hash"))
    try:
        max_concurrency = max(1, int(config.get("max_concurrency", 3)))
    except (TypeError, ValueError):
        max_concurrency = 3
    semaphore = asyncio.Semaphore(max_concurrency)
    stats = RunStats()
    await process_pending_items(
        model,
        prompt,
        pending_store,
        results_store,
        processed_hashes,
        progress_store,
        stats,
        logger,
    )
    client = TelegramClient(str(session_path), api_id, api_hash)
    try:
        await client.start(phone=phone_number)
    except errors.SessionPasswordNeededError:
        logger.error("Two-factor authentication required; update the session manually")
        await client.disconnect()
        return
    except Exception as exc:
        logger.error("Failed to start Telegram client: %s", exc)
        await client.disconnect()
        return
    try:
        for channel in channels:
            last_id = progress_store.data.get("channels", {}).get(channel, 0)
            try:
                images, new_last_id = await scrape_channel(
                    client,
                    channel,
                    start_date,
                    last_id,
                    TEMP_DIR,
                    logger,
                )
            except Exception as exc:
                logger.exception("Failed to scrape %s: %s", channel, exc)
                continue
            if images:
                await process_images(
                    images,
                    model,
                    prompt,
                    results_store,
                    progress_store,
                    pending_store,
                    processed_hashes,
                    semaphore,
                    stats,
                    logger,
                )
            if new_last_id > last_id:
                await progress_store.mutate(
                    lambda data, ch=channel, mid=new_last_id: _update_channel_progress(data, ch, mid)
                )
    finally:
        await client.disconnect()
    logger.info("Run complete: %s", stats.snapshot())


def _update_channel_progress(data: Dict[str, Any], channel: str, message_id: int) -> None:
    channels = data.setdefault("channels", {})
    channels[channel] = max(message_id, channels.get(channel, 0))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram scraping and categorisation agent")
    parser.add_argument("--run-once", action="store_true", help="Execute the pipeline immediately and exit")
    return parser.parse_args()


async def main() -> None:
    args = parse_arguments()
    ensure_directories()
    logger = setup_logging()
    if args.run_once:
        await run_pipeline_job(logger)
        return
    try:
        config = load_config(CONFIG_PATH)
    except ConfigError as err:
        logger.warning("Configuration error (%s); scheduler will still initialise", err)
        config = {
            "daily_run_time": "02:00",
        }
    hour, minute = parse_daily_time(config.get("daily_run_time"), logger)
    scheduler = AsyncIOScheduler(timezone=datetime.now().astimezone().tzinfo)
    scheduler.add_job(
        run_pipeline_job,
        trigger="cron",
        hour=hour,
        minute=minute,
        kwargs={"logger": logger},
    )
    scheduler.add_job(
        run_pipeline_job,
        trigger="date",
        run_date=datetime.now().astimezone(),
        kwargs={"logger": logger},
    )
    scheduler.start()
    logger.info("Scheduler active; daily run at %02d:%02d", hour, minute)
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
