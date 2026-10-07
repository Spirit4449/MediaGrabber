"""Bounded, offset-based document downloads using Telethon's public API."""
import asyncio
import os

CHUNK_SIZE = 512 * 1024


async def download_document(client, media, target, size, workers, progress):
    chunks = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
    workers = min(workers, chunks)
    completed = 0
    # All writes are synchronous on the event loop: seek/write cannot interleave.
    with open(target, "wb") as output:
        async def transfer(index):
            nonlocal completed
            start = chunks * index // workers
            end = chunks * (index + 1) // workers
            offset = start * CHUNK_SIZE
            stream = client.iter_download(
                media, offset=offset, limit=end - start,
                request_size=CHUNK_SIZE, chunk_size=CHUNK_SIZE, file_size=size,
            )
            try:
                async for data in stream:
                    expected = min(CHUNK_SIZE, size - offset)
                    if offset >= end * CHUNK_SIZE or len(data) != expected:
                        raise IOError("Unexpected Telegram download chunk size")
                    output.seek(offset)
                    output.write(data)
                    offset += len(data)
                    completed += len(data)
                    progress(completed, size)
                if offset != min(end * CHUNK_SIZE, size):
                    raise IOError("Incomplete Telegram download range")
            finally:
                await stream.close()

        tasks = [asyncio.create_task(transfer(i)) for i in range(workers)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    if completed != size or os.path.getsize(target) != size:
        raise IOError("Incomplete Telegram download")
    return target
