import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import moments_sync as moments
from reliability import load_state

ZONE = ZoneInfo("America/New_York")


def message(id, text, when, **extra):
    return SimpleNamespace(id=id, message=text, date=when, edit_date=extra.get("edit_date"),
                           grouped_id=extra.get("grouped_id"), photo=extra.get("photo", object()))


class FakeSite:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.imports = []
        self.logins = 0

    def login(self, username, password):
        self.logins += 1

    def import_posts(self, channel_id, username, posts, complete):
        self.imports.append({"channel": channel_id, "username": username, "posts": posts, "complete": complete})
        return {"inserted": len(posts), "updated": 0, "unchanged": 0}

    def status(self, ids):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


class MomentsSyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.state_file = Path(self.folder.name) / "moments.json"
        now = datetime.now(timezone.utc)
        self.today = message(41, "Swamishri asked twice whether the boy was hurt.", now, edit_date=now)
        self.messages = [self.today, message(40, "", now - timedelta(days=1)),  # album companion, no text
                         message(39, "An older moment.", now - timedelta(days=2), photo=None)]

    def tearDown(self):
        self.folder.cleanup()

    def args(self, **overrides):
        values = dict(channel=2074676830, state_file=str(self.state_file), session="unused", dry_run=False, out="",
                      limit=0, wait=5, poll=0, stop_after_today=True)
        values.update(overrides)
        return SimpleNamespace(**values)

    def client(self):
        async def iter_messages(entity, limit=None):
            for item in self.messages[:limit] if limit else self.messages:
                yield item
        return SimpleNamespace(connect=AsyncMock(), disconnect=AsyncMock(), is_user_authorized=AsyncMock(return_value=True),
                               get_entity=AsyncMock(return_value=SimpleNamespace(id=2074676830, username="MomentsWithMS")), iter_messages=iter_messages)

    async def test_full_read_is_sent_then_waits_for_todays_post(self):
        site = FakeSite([[{"messageId": 41, "status": "processing"}],
                         [{"messageId": 41, "status": "done", "title": "Asked twice", "link": "linked",
                           "linkedTitle": "The limp", "linkedRecording": "Report 2026-10-15"}]])
        client = self.client()
        code, notify = await self._run(site, self.args(), client)
        self.assertEqual(code, 0)
        sent = site.imports[0]
        self.assertTrue(sent["complete"])
        self.assertEqual(sent["username"], "MomentsWithMS")
        self.assertEqual([p["messageId"] for p in sent["posts"]], [39, 41])  # empty-text message left out
        self.assertTrue(sent["posts"][1]["editedAt"])
        self.assertFalse(sent["posts"][0]["hasPhoto"])
        self.assertEqual(load_state(self.state_file)["last_delivery_date"], datetime.now(ZONE).date().isoformat())
        self.assertIn("retold as one", notify.call_args.args[2])
        # Once delivered, later checks today do not touch Telegram or the site.
        code, _ = await self._run(site, self.args(), client)
        self.assertEqual(code, 0)
        self.assertEqual(client.connect.await_count, 1)
        self.assertEqual(len(site.imports), 1)

    async def _run(self, site, args, client=None):
        client = client or self.client()
        with patch.dict("os.environ", {"TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "x"}), \
             patch.object(moments, "TelegramClient", return_value=client), patch.object(moments, "load_download_session"), \
             patch.object(moments, "resolve_entity_safely", new=AsyncMock(return_value=object())), \
             patch.object(moments, "_resolve_bns_admin_credentials", return_value=("admin", "pw")), \
             patch.object(moments, "BnsSite", return_value=site), patch.object(moments, "notify") as notify:
            return await moments.run_sync(args), notify

    async def test_no_post_today_keeps_checking(self):
        self.messages = self.messages[1:]
        site = FakeSite([[]])
        code, _ = await self._run(site, self.args())
        self.assertEqual(code, 3)
        self.assertEqual(load_state(self.state_file), {})
        self.assertEqual(len(site.imports), 1)  # older posts and edits are still sent

    async def test_unprocessed_post_raises_for_retry_without_marking_delivery(self):
        site = FakeSite([[{"messageId": 41, "status": "error", "error": "AI capacity is used up"}]])
        with self.assertRaisesRegex(RuntimeError, "AI capacity"):
            await self._run(site, self.args(wait=0))
        self.assertEqual(load_state(self.state_file), {})

    async def test_skipped_post_counts_as_delivered(self):
        site = FakeSite([[{"messageId": 41, "status": "skipped", "skipReason": "A schedule"}]])
        code, notify = await self._run(site, self.args())
        self.assertEqual(code, 0)
        self.assertIn("skipped: A schedule", notify.call_args.args[2])

    async def test_limit_never_claims_a_complete_read(self):
        site = FakeSite([[{"messageId": 41, "status": "done", "title": "T", "link": "candidate", "linkedTitle": "L"}]])
        code, notify = await self._run(site, self.args(limit=1))
        self.assertFalse(site.imports[0]["complete"])
        self.assertIn("links.mjs --show 41", notify.call_args.args[2])

    async def test_dry_run_writes_posts_and_sends_nothing(self):
        out = Path(self.folder.name) / "posts.jsonl"
        site = FakeSite([[]])
        code, _ = await self._run(site, self.args(dry_run=True, out=str(out), stop_after_today=False))
        self.assertEqual(code, 0)
        self.assertEqual(site.imports, [])
        self.assertEqual(len(out.read_text().splitlines()), 2)

    async def test_unseen_channel_lists_the_channels_the_account_can_see(self):
        dialogs = [SimpleNamespace(entity=SimpleNamespace(id=555), name="Moments with Mahant Swami", is_channel=True),
                   SimpleNamespace(entity=SimpleNamespace(id=777), name="Family", is_channel=False, is_group=True)]
        client = SimpleNamespace(get_dialogs=AsyncMock(return_value=dialogs))
        with patch.object(moments, "resolve_entity_safely", new=AsyncMock(side_effect=ValueError("not found"))):
            with self.assertRaisesRegex(RuntimeError, r"cannot see channel 2074676830[\s\S]*look related:\n  555  Moments with Mahant Swami$"):
                await moments.resolve_channel(client, 2074676830)

    def test_channel_ref_accepts_ids_usernames_and_links(self):
        self.assertEqual(moments.channel_ref("2074676830"), 2074676830)
        self.assertEqual(moments.channel_ref("-1002074676830"), 2074676830)
        self.assertEqual(moments.channel_ref("@MomentsWithMS"), "MomentsWithMS")
        self.assertEqual(moments.channel_ref("https://t.me/MomentsWithMS"), "MomentsWithMS")

    def test_post_payload(self):
        when = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
        self.assertIsNone(moments.post_payload(message(1, "   ", when)))
        payload = moments.post_payload(message(2, "Text", when, grouped_id=99))
        self.assertEqual(payload, {"messageId": 2, "postedAt": "2026-10-16T12:00:00+00:00", "editedAt": "", "text": "Text",
                                   "groupedId": "99", "hasPhoto": True})
        # 8 a.m. New York is still "today" there even though it is a different UTC hour.
        self.assertEqual(moments.local_date("2026-10-17T03:30:00+00:00", ZONE), "2026-10-16")


class BnsSiteTests(unittest.TestCase):
    def test_login_cookie_is_sent_with_import_and_errors_are_readable(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, self.headers.get("Cookie"), body))
                if self.path == "/api/admin/login":
                    self.send_response(200)
                    self.send_header("Set-Cookie", "bns_session=abc; Path=/; HttpOnly")
                    self.end_headers()
                    self.wfile.write(json.dumps({"isAdmin": body["username"] == "admin"}).encode())
                elif self.headers.get("Cookie") != "bns_session=abc":
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b'{"error":"Admin login required."}')
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps({"inserted": len(body["posts"])}).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with self.assertRaisesRegex(RuntimeError, "Admin login required"):
                moments.BnsSite(base).import_posts(1, "", [], True)
            with self.assertRaisesRegex(RuntimeError, "not an admin"):
                moments.BnsSite(base).login("viewer", "pw")
            site = moments.BnsSite(base)
            site.login("admin", "pw")
            self.assertEqual(site.import_posts(1, "u", [{"messageId": 1}], True), {"inserted": 1})
            self.assertEqual(seen[-1][1], "bns_session=abc")
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
