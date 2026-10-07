const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const express = require('express');
const { registerApi } = require('../src/api');

async function fixture(t) {
  const rootDir = await fs.mkdtemp(path.join(os.tmpdir(), 'media-api-'));
  t.after(() => fs.rm(rootDir, { recursive: true, force: true }));
  const app = express(), starts = [], messages = [];
  app.use(express.json({ limit: '256kb' }));
  const telegram = { sendMessage: async (...args) => { messages.push(args); return { message_id: 42 }; } };
  registerApi(app, { bot: { telegram }, sharedSecret: 'test-secret', rootDir,
    startBackgroundDownload: (...args) => starts.push(args) });
  const server = await new Promise((resolve, reject) => {
    const listener = app.listen(0, '127.0.0.1', error => error ? reject(error) : resolve(listener));
    listener.once('error', reject);
  });
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (body, sign = true) => fetch(`${base}/api/download`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-signature': sign
      ? crypto.createHmac('sha256', 'test-secret').update(JSON.stringify(body)).digest('hex') : 'invalid' },
    body: JSON.stringify(body),
  });
  return { base, post, starts, messages, rootDir, telegram };
}

test('API checks signatures and links before starting one background job', async t => {
  const f = await fixture(t);
  const body = { link: 'https://t.me/example/12', chat_id: 123, caption: 'caption', forceVideo: true };
  assert.equal((await f.post(body, false)).status, 401);
  assert.equal((await f.post({ ...body, link: 'https://example.com/file' })).status, 400);
  assert.equal((await f.post({ link: body.link })).status, 400);
  assert.equal(f.starts.length, 0);
  assert.equal(f.messages.length, 0);
  const response = await f.post(body);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ok: true });
  assert.equal(f.starts.length, 1);
  assert.deepEqual(f.starts[0], [{ chat: { id: 123 }, telegram: f.telegram }, body.link, 42,
    { caption: 'caption', forceVideo: true }]);
  assert.ok((await fs.stat(path.join(f.rootDir, 'downloads'))).isDirectory());
  assert.deepEqual(await (await fetch(`${f.base}/healthz`)).json(), { ok: true });
});

test('API does not start a worker when the initial Telegram message fails', async t => {
  const f = await fixture(t);
  f.telegram.sendMessage = async () => { throw new Error('Telegram unavailable'); };
  assert.equal((await f.post({ link: 'https://t.me/example/1', chat_id: 123 })).status, 500);
  assert.equal(f.starts.length, 0);
});
