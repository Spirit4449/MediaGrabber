import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import daily_bns_sync as bns
from reliability import load_state, save_state, exclusive_lock


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_website_failure_blocks_checkpoint_then_retry_saves_completed_step(self):
        with tempfile.TemporaryDirectory() as folder:
            state_file = Path(folder) / 'state.json'
            save_state(state_file, {'last_id': 10})
            args = SimpleNamespace(session='unused', audio_only=True, no_site_upload=False,
                                   state_file=str(state_file), source=1, target=2, seed=False,
                                   only_message_id=0, site_upload_timeout=60)
            async def download(file):
                target = Path(file) / 'audio.mp3'
                target.write_bytes(b'audio')
                return str(target)
            msg = SimpleNamespace(id=11, media=True, message='caption', date=datetime.now(timezone.utc),
                                  download_media=AsyncMock(side_effect=download))
            later = SimpleNamespace(id=12, media=False)
            async def messages(*args, **kwargs):
                for item in [msg, later]:
                    yield item
            client = SimpleNamespace(connect=AsyncMock(), is_user_authorized=AsyncMock(return_value=True),
                                     disconnect=AsyncMock(), iter_messages=messages,
                                     send_file=AsyncMock(side_effect=[RuntimeError('send failed'), None]))
            with patch.dict('os.environ', {'TELEGRAM_API_ID': '1', 'TELEGRAM_API_HASH': 'test',
                                          'BNS_DISABLE_SITE_UPLOAD': 'false'}), \
                 patch.object(bns, 'TelegramClient', return_value=client), \
                 patch.object(bns, 'load_download_session'), \
                 patch.object(bns, '_resolve_bns_admin_credentials', return_value=('user', 'pass')), \
                 patch.object(bns, 'resolve_entity_safely', new=AsyncMock(return_value=object())), \
                 patch.object(bns, 'is_audio_message', return_value=True), \
                 patch.object(bns, '_ext_from_media', return_value='.mp3'), \
                 patch.object(bns, 'convert_audio_to_mp3', side_effect=lambda p: p), \
                 patch.object(bns, 'notify'), \
                 patch.object(bns, 'upload_to_bns_site', side_effect=[{'status': 'error'}, {'status': 'uploaded'}]) as upload:
                with self.assertRaisesRegex(RuntimeError, 'Website upload failed'):
                    await bns.run_sync(args)
                self.assertEqual(load_state(state_file)['last_id'], 10)
                client.send_file.assert_not_awaited()
                with self.assertRaisesRegex(RuntimeError, 'send failed'):
                    await bns.run_sync(args)
                self.assertTrue(load_state(state_file)['pending']['11']['website'])
                args.stop_after_today = True
                await bns.run_sync(args)
                self.assertEqual(upload.call_count, 2)  # Third attempt reused completed website step.
                self.assertEqual(load_state(state_file)['last_id'], 11)  # Stop before later post.
                await bns.run_sync(args)  # A restart must not reconnect after today's success.
                self.assertEqual(client.connect.await_count, 3)
                self.assertEqual(load_state(state_file)['pending'], {})
                self.assertEqual(client.disconnect.await_count, 3)

    async def test_empty_download_does_not_advance_state(self):
        with tempfile.TemporaryDirectory() as folder:
            state_file = Path(folder) / 'state.json'
            save_state(state_file, {'last_id': 10})
            args = SimpleNamespace(session='unused', audio_only=False, no_site_upload=True,
                                   state_file=str(state_file), source=1, target=2, seed=False, only_message_id=0)
            async def messages(*args, **kwargs):
                yield SimpleNamespace(id=11, media=True, download_media=AsyncMock(return_value=None))
                yield SimpleNamespace(id=12, media=False)
            client = SimpleNamespace(connect=AsyncMock(), is_user_authorized=AsyncMock(return_value=True),
                                     disconnect=AsyncMock(), iter_messages=messages)
            with patch.dict('os.environ', {'TELEGRAM_API_ID': '1', 'TELEGRAM_API_HASH': 'test'}), \
                 patch.object(bns, 'TelegramClient', return_value=client), \
                 patch.object(bns, 'load_download_session'), \
                 patch.object(bns, '_resolve_bns_admin_credentials', return_value=('', '')), \
                 patch.object(bns, 'resolve_entity_safely', new=AsyncMock(return_value=object())):
                with self.assertRaisesRegex(RuntimeError, 'retained for retry'):
                    await bns.run_sync(args)
            self.assertEqual(load_state(state_file), {'last_id': 10})

    def test_corrupt_state_is_not_silently_reseeded_and_lock_prevents_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            path.write_text('{broken')
            with self.assertRaises(ValueError):
                load_state(path)
            with exclusive_lock(str(path) + '.lock'):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(str(path) + '.lock'):
                        self.fail('Concurrent run was allowed')
            with exclusive_lock(str(path) + '.lock'):
                save_state(path, {'last_id': 12})
            self.assertEqual(load_state(path), {'last_id': 12})

    async def test_backlog_is_drained_oldest_first_across_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            state_file = Path(folder) / 'state.json'
            save_state(state_file, {'last_id': 10})
            args = SimpleNamespace(session='unused', audio_only=False, no_site_upload=True,
                                   state_file=str(state_file), source=1, target=2, seed=False, only_message_id=0)
            seen = []
            async def messages(source, min_id, reverse, limit):
                self.assertTrue(reverse)
                for message_id in list(range(min_id + 1, 262))[:limit]:
                    seen.append(message_id)
                    yield SimpleNamespace(id=message_id, media=False)
            client = SimpleNamespace(connect=AsyncMock(), is_user_authorized=AsyncMock(return_value=True),
                                     disconnect=AsyncMock(), iter_messages=messages)
            with patch.dict('os.environ', {'TELEGRAM_API_ID': '1', 'TELEGRAM_API_HASH': 'test'}), \
                 patch.object(bns, 'TelegramClient', return_value=client), \
                 patch.object(bns, 'load_download_session'), \
                 patch.object(bns, '_resolve_bns_admin_credentials', return_value=('', '')), \
                 patch.object(bns, 'resolve_entity_safely', new=AsyncMock(return_value=object())), \
                 patch('builtins.print'):
                await bns.run_sync(args)
                self.assertEqual(load_state(state_file)['last_id'], 210)
                await bns.run_sync(args)
                self.assertEqual(load_state(state_file)['last_id'], 261)
            self.assertEqual(seen, list(range(11, 262)))
