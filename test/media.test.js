const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { createMediaDelivery } = require('../src/media-delivery');
const { createSharedDownloads } = require('../src/shared-downloads');
const { createCompressor, geometry, run } = require('../src/media-compression');

async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'media-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  return root;
}
test('publishes exact original bytes and cleans expired files only', async t => {
  const root = await fixture(t);
  const source = path.join(root, 'original file.bin');
  const bytes = Buffer.from([0, 1, 2, 255]);
  await fs.writeFile(source, bytes);
  const store = createSharedDownloads({ directory: path.join(root, 'public'), baseUrl: 'https://bns.classchats.net/downloads/', retentionHours: 72 });
  const a = await store.publish(source), b = await store.publish(source);
  assert.notEqual(a.path, b.path);
  assert.deepEqual(await fs.readFile(a.path), bytes);
  assert.match(a.url, /^https:\/\/bns.classchats.net\/downloads\//);
  const unrelated = path.join(store.root, 'keep.txt');
  await fs.writeFile(unrelated, 'keep');
  // Recreate module to model an app restart; expiry uses file timestamps.
  const restarted = createSharedDownloads({ directory: store.root, baseUrl: a.url, retentionHours: 72 });
  await restarted.cleanup(Date.now() + 73 * 3600000);
  await assert.rejects(fs.stat(a.path), { code: 'ENOENT' });
  assert.equal(await fs.readFile(unrelated, 'utf8'), 'keep');
  assert.deepEqual(await fs.readFile(source), bytes);
});
test('missing public URL does not discard the original', async t => {
  const root = await fixture(t), source = path.join(root, 'original');
  await fs.writeFile(source, 'original');
  const store = createSharedDownloads({ directory: root });
  await assert.rejects(store.publish(source), /PUBLIC_DOWNLOAD_BASE_URL/);
  assert.equal(await fs.readFile(source, 'utf8'), 'original');
});
test('oversized delivery sends only the original link and preserves source on message failure', async t => {
  const root = await fixture(t), source = path.join(root, 'video.mp4');
  const store = createSharedDownloads({ directory: path.join(root, 'public'), baseUrl: 'https://bns.classchats.net/downloads/' });
  const deliver = createMediaDelivery({ maxUploadMb: 0.000001, sharedDownloads: store,
    compressor: { compress() { throw new Error('Should not compress'); } } });
  await fs.writeFile(source, 'original bytes');
  let message;
  await deliver({ sendMessage: async (...args) => { message = args; } }, 1, source);
  assert.match(message[2].reply_markup.inline_keyboard[0][0].url, /^https:/);
  await assert.rejects(fs.stat(source), { code: 'ENOENT' });
  await fs.writeFile(source, 'retry original');
  await assert.rejects(deliver({ sendMessage: async () => { throw new Error('Telegram down'); } }, 1, source), /Telegram down/);
  assert.equal(await fs.readFile(source, 'utf8'), 'retry original');
});
test('real encoding preserves landscape, portrait, anamorphic and rotated display ratios', async t => {
  try { await run('ffmpeg', ['-version']); await run('ffprobe', ['-version']); }
  catch { t.skip('FFmpeg and FFprobe are required for integration tests'); return; }
  const root = await fixture(t);
  const compressor = createCompressor({ targetMb: 1 });
  for (const [name, size, sar, rotated] of [
    ['landscape', '640x360', '1', false], ['square', '480x480', '1', false], ['portrait', '360x640', '1', false],
    ['anamorphic', '640x480', '4/3', false], ['rotated', '640x360', '1', true],
  ]) {
    let source = path.join(root, `${name}.mp4`);
    await run('ffmpeg', ['-y', '-v', 'error', '-f', 'lavfi', '-i', `testsrc2=size=${size}:rate=12:duration=1`, '-vf', `setsar=${sar}`, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', source]);
    if (rotated) {
      const output = path.join(root, 'rotation.mp4');
      await run('ffmpeg', ['-y', '-v', 'error', '-i', source, '-c', 'copy', '-metadata:s:v:0', 'rotate=90', output]);
      source = output;
    }
    const original = await fs.readFile(source);
    const before = await compressor.probe(source);
    const result = await compressor.compress(source, { isVideo: true, maxBytes: 1024 * 1024 });
    assert.ok(result.path, `${name}: ${result.error}`);
    const after = await compressor.probe(result.path);
    const inputRatio = geometry(before.streams[0]).ratio;
    assert.ok(Math.abs(geometry(after.streams[0]).ratio / inputRatio - 1) < 0.01, name);
    assert.equal(after.streams[0].sample_aspect_ratio, '1:1');
    assert.ok((await fs.stat(result.path)).size <= 1024 * 1024);
    assert.deepEqual(await fs.readFile(source), original);
    await result.cleanup();
  }
});
test('long media rejects an unusable size budget and audio converts once from original', async t => {
  try { await run('ffmpeg', ['-version']); } catch { t.skip('FFmpeg required'); return; }
  const root = await fixture(t), source = path.join(root, 'sound.wav');
  await run('ffmpeg', ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', source]);
  const compressor = createCompressor();
  const impossible = await compressor.compress(source, { isAudio: true, maxBytes: 1024 });
  assert.match(impossible.error, /quality loss/);
  const result = await compressor.compress(source, { isAudio: true, maxBytes: 1024 * 1024 });
  assert.ok(result.path, result.error);
  assert.equal((await compressor.probe(result.path)).streams[0].codec_name, 'mp3');
  await result.cleanup();
});
