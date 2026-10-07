import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TELEGRAM_API_ID', '1')
os.environ.setdefault('TELEGRAM_API_HASH', 'test')
import downloader


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_failure_falls_back_and_overwrites_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = str(Path(folder) / 'file.bin')
            async def parallel(*args):
                Path(target).write_bytes(b'partial junk')
                raise IOError('network failure')
            async def sequential(**kwargs):
                Path(kwargs['file']).write_bytes(b'original')
                kwargs['progress_callback'](8, 8)
                return kwargs['file']
            msg = SimpleNamespace(document=SimpleNamespace(size=9 * 1024 * 1024),
                                  media=object(), download_media=AsyncMock(side_effect=sequential))
            with patch.object(downloader, '_derive_target_path', return_value=target), \
                 patch.object(downloader, 'download_document', side_effect=parallel), \
                 patch.object(downloader, 'emit') as emit:
                result = await downloader.download_with_progress(msg, folder, asyncio.Event())
            self.assertEqual(Path(result).read_bytes(), b'original')
            msg.download_media.assert_awaited_once()
            self.assertTrue(any(call.args[0] == 'status' for call in emit.call_args_list))

    async def test_cancellation_removes_partial_file_without_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            target = str(Path(folder) / 'file.bin')
            async def parallel(*args):
                Path(target).write_bytes(b'partial')
                raise asyncio.CancelledError
            msg = SimpleNamespace(document=SimpleNamespace(size=9 * 1024 * 1024),
                                  media=object(), download_media=AsyncMock())
            with patch.object(downloader, '_derive_target_path', return_value=target), \
                 patch.object(downloader, 'download_document', side_effect=parallel):
                with self.assertRaises(asyncio.CancelledError):
                    await downloader.download_with_progress(msg, folder, asyncio.Event())
            self.assertFalse(Path(target).exists())
            msg.download_media.assert_not_awaited()

    async def test_cached_entity_skips_dialog_list_and_private_miss_refreshes(self):
        for cached in (True, False):
            client = SimpleNamespace(get_input_entity=AsyncMock(side_effect=[object()] if cached else [ValueError(), object()]),
                                     get_dialogs=AsyncMock())
            with patch.object(downloader, 'client', client):
                await downloader.resolve_entity('peer_id', -100123)
            self.assertEqual(client.get_dialogs.await_count, 0 if cached else 1)
