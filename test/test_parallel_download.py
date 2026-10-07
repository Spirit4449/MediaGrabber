import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parallel_download import CHUNK_SIZE, download_document


class FakeStream:
    def __init__(self, client, offset, limit):
        self.client, self.offset, self.limit = client, offset, limit
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.limit:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        if self.client.fail and self.offset == 0:
            raise IOError("network failure")
        data = self.client.data[self.offset:self.offset + CHUNK_SIZE]
        self.offset += CHUNK_SIZE
        self.limit -= 1
        return data

    async def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, data, fail=False):
        self.data, self.fail, self.streams = data, fail, []

    def iter_download(self, media, **kwargs):
        stream = FakeStream(self, kwargs['offset'], kwargs['limit'])
        self.streams.append(stream)
        return stream


class ParallelDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_bytes_and_progress_for_partial_and_full_chunks(self):
        for size in (CHUNK_SIZE * 17 + 123, CHUNK_SIZE * 16, 123):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as folder:
                data = os.urandom(size)
                client = FakeClient(data)
                progress = []
                target = str(Path(folder) / 'file')
                await download_document(client, object(), target, size, 4,
                                        lambda done, total: progress.append((done, total)))
                self.assertEqual(Path(target).read_bytes(), data)
                self.assertEqual(progress[-1], (size, size))
                self.assertEqual([p[0] for p in progress], sorted(p[0] for p in progress))
                self.assertTrue(all(stream.closed for stream in client.streams))

    async def test_failure_cancels_and_closes_all_streams(self):
        with tempfile.TemporaryDirectory() as folder:
            client = FakeClient(b'x' * CHUNK_SIZE * 20, fail=True)
            with self.assertRaisesRegex(IOError, 'network failure'):
                await download_document(client, object(), str(Path(folder) / 'file'),
                                        len(client.data), 4, lambda *args: None)
            self.assertTrue(all(stream.closed for stream in client.streams))

    async def test_short_read_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            client = FakeClient(b'x' * 100)
            with self.assertRaises(IOError):
                await download_document(client, object(), str(Path(folder) / 'file'),
                                        CHUNK_SIZE * 2, 2, lambda *args: None)


if __name__ == '__main__':
    unittest.main()
