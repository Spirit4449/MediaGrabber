const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { createDownloadFlow } = require('../src/download-flow');

async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'download-flow-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const spawns = [], edits = [], replies = [];
  let handlers;
  const flow = createDownloadFlow({
    rootDir: root, pythonBin: 'python3',
    safeUploadAndDelete: async () => {},
    spawnWorker: args => { spawns.push(args); return { stderr: { on() {} } }; },
    consumeWorker: (child, callbacks) => { handlers = callbacks; },
  });
  const ctx = { chat: { id: 123 }, reply: async text => { replies.push(text); return { message_id: 9 }; },
    telegram: { editMessageText: async (...args) => edits.push(args[3]), deleteMessage: async () => {} } };
  return { flow, ctx, root, spawns, edits, replies, state: flow.state, event: ev => handlers.onEvent(ev) };
}

test('bot uses one worker for access check and download and preserves invite retry', async t => {
  const f = await fixture(t), link = 'https://t.me/c/123/456';
  await f.flow.handleDownloadFlow(f.ctx, link);
  assert.equal(f.spawns.length, 1);
  assert.equal(f.spawns[0].includes('--preflight'), false);
  await f.event({ type: 'need_invite' });
  assert.equal(f.state.get(123).awaitingInviteFor, link);
  assert.match(f.edits[0], /invite link/);
  await f.flow.handleDownloadFlow(f.ctx, link, { invite: 'hash' });
  assert.equal(f.spawns.length, 2);
  assert.equal(f.spawns[1].at(-1), 'hash');
});

test('bot throttles progress while errors are delivered immediately', async t => {
  const f = await fixture(t);
  await f.flow.handleDownloadFlow(f.ctx, 'https://t.me/example/1');
  for (let pct = 1; pct <= 10; pct++) await f.event({ type: 'progress', pct });
  assert.equal(f.edits.length, 1);
  await f.event({ type: 'error', text: 'No media in this post' });
  assert.match(f.edits.at(-1), /No media/);
});

test('API and bot jobs use separate directories below the configured project', async t => {
  const f = await fixture(t);
  await f.flow.handleDownloadFlow(f.ctx, 'https://t.me/example/1');
  f.flow.startBackgroundDownload(f.ctx, 'https://t.me/example/2', 10);
  const directories = f.spawns.map(args => args[args.indexOf('--outdir') + 1]);
  assert.notEqual(directories[0], directories[1]);
  for (const directory of directories) assert.equal(path.dirname(directory), path.join(f.root, 'downloads'));
  await f.event({ type: 'need_invite' });
  assert.equal(f.state.has(123), false);
  assert.match(f.edits.at(-1), /No access/);
});
