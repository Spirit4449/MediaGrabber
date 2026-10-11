"""PM2 entry point: supervise bot polling or retry a bounded BNS sync.

Uses only the standard library and the project's existing python-dotenv.
"""
import argparse
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, time as wall_time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from reliability import exclusive_lock, load_state, save_state

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def terminate_group(child):
    # Also clean up downloader/ffmpeg/Chrome descendants after their parent exits.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def polling_stale(file, pid, started, timeout, now=None):
    now = time.time() if now is None else now
    try:
        health = load_state(file)
        last = float(health["polled_at"]) if health.get("pid") == pid else started
    except (OSError, ValueError, KeyError, TypeError):
        last = started
    return now - max(started, min(last, now)) > timeout


def run_child(command, timeout, health_file=None, tick=None):
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL)
    started = time.time()
    started_mono = time.monotonic()
    try:
        while child.poll() is None:
            if health_file:
                expired = polling_stale(health_file, child.pid, started, timeout)
            else:
                expired = time.monotonic() - started_mono > timeout
            if expired:
                print("[watchdog] No polling progress or job exceeded its time limit", flush=True)
                return 124
            if tick:
                tick()
            time.sleep(1)
        return child.returncode
    finally:
        terminate_group(child)


def check_deadline(state_file, alert_file, now, deadline, send):
    today = now.date().isoformat()
    if now.time() < deadline:
        return
    state = load_state(state_file)
    alerts = load_state(alert_file)
    if state.get("last_delivery_date", "") >= today or alerts.get("date") == today:
        return
    # Record only after successful notification so a temporary network failure retries.
    send(f"[BNS missing delivery] {state_file.name}: no completed source post dated {today}. "
         "Scheduled checks have stopped for today; check the worker logs.")
    save_state(alert_file, {"date": today})


def should_check(state_file, now, start_at, deadline):
    return (start_at <= now.time() < deadline
            and load_state(state_file).get("last_delivery_date", "") < now.date().isoformat())


def positive_env(name, default):
    value = float(os.environ.get(name, default))
    if not 0 < value < float("inf"):
        raise ValueError(f"{name} must be positive and finite")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["bot", "bns"])
    parser.add_argument("--state-file", default=".state/bns_state.json")
    parser.add_argument("--start-at", default="00:00")
    parser.add_argument("--target", type=int)
    parser.add_argument("--audio-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)

    def stop(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if args.mode == "bot":
        health = Path(os.environ.get("BOT_HEALTH_FILE", str(ROOT / ".state/bot-health.json")))
        os.environ["BOT_HEALTH_FILE"] = str(health.resolve())
        with exclusive_lock(ROOT / ".state/bot-worker.lock"):
            return run_child([os.environ.get("NODE_BIN", "node"), str(ROOT / "server.js")],
                             positive_env("BOT_POLL_TIMEOUT_SECONDS", "300"), health_file=health)

    interval = positive_env("BNS_CHECK_INTERVAL_SECONDS", "900")
    timeout = positive_env("BNS_RUN_TIMEOUT_SECONDS", "1800")
    zone = ZoneInfo(os.environ.get("BNS_TIMEZONE", "America/New_York"))
    start_at = wall_time.fromisoformat(args.start_at)
    deadline = wall_time.fromisoformat(os.environ.get("BNS_ALERT_AFTER", "18:00"))
    if start_at >= deadline:
        raise ValueError("BNS start time must be before its daily cutoff")
    state_file = Path(args.state_file).resolve()
    alert_file = Path(str(state_file) + ".alert.json")
    command = [sys.executable, "-u", str(ROOT / "daily_bns_sync.py"), "--state-file", str(state_file),
               "--stop-after-today"]
    if args.target is not None:
        command += ["--target", str(args.target)]
    if args.audio_only:
        command += ["--audio-only"]
    token = os.environ.get("BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID")
    last_alert_check = 0

    def tick():
        nonlocal last_alert_check
        if time.monotonic() - last_alert_check < 60:
            return
        last_alert_check = time.monotonic()
        if token and chat:
            from daily_bns_sync import send_telegram_bot_message
            try:
                check_deadline(state_file, alert_file, datetime.now(zone), deadline,
                               lambda text: send_telegram_bot_message(token, chat, text))
            except Exception as exc:
                print(f"[alert failed] {exc}", flush=True)

    if not token or not chat:
        print("[worker] Missing BOT_TOKEN/TELEGRAM_NOTIFY_CHAT_ID; deadline alerts disabled", flush=True)
    with exclusive_lock(str(state_file) + ".worker.lock"):
        while True:
            # Reading local state is cheap; do not contact Telegram after success/cutoff.
            if should_check(state_file, datetime.now(zone), start_at, deadline):
                code = run_child(command, timeout, tick=tick)
                print(f"[worker] BNS exited {code}; next schedule evaluation in {interval:g}s", flush=True)
            end = time.monotonic() + interval
            while time.monotonic() < end:
                tick()
                time.sleep(min(1, max(0, end - time.monotonic())))


if __name__ == "__main__":
    sys.exit(main())
