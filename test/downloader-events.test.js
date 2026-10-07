const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { PassThrough } = require('node:stream');
const { consumeDownloader } = require('../downloader-events');
function worker() { const child = new EventEmitter(); child.stdout = new PassThrough(); return child; }
test('completion follows progress in order; late progress cannot overwrite delivery', async () => {
  const child = worker(), seen = [];
  const consumer = consumeDownloader(child, {
    onEvent: async ev => { await new Promise(resolve => setImmediate(resolve)); seen.push(ev.type); },
    onFailure: async error => { throw error; }, onLog() {},
  });
  child.stdout.write('{"type":"progress"}\n{"type":"do');
  child.stdout.write('ne","path":"file"}\n{"type":"progress"}\n');
  child.emit('close', 0);
  await consumer.settled();
  assert.deepEqual(seen, ['progress', 'done']);
});
test('clean exit with no completion becomes a visible error; unterminated final event is parsed', async () => {
  for (const withDone of [false, true]) {
    const child = worker(), failures = [], events = [];
    const consumer = consumeDownloader(child, { onEvent: async ev => events.push(ev.type), onFailure: async error => failures.push(error.message), onLog() {} });
    child.stdout.write(withDone ? '{"type":"done","path":"file"}' : '{"type":"progress"}\n');
    child.emit('close', 0);
    await consumer.settled();
    assert.equal(failures.length, withDone ? 0 : 1);
    if (!withDone) assert.match(failures[0], /without a completion event/);
    else assert.deepEqual(events, ['done']);
  }
});
test('spawn errors and handler errors are reported', async () => {
  for (const spawnError of [true, false]) {
    const child = worker(), failures = [];
    const consumer = consumeDownloader(child, { onEvent: async () => { throw new Error('handler failed'); }, onFailure: async error => failures.push(error.message), onLog() {} });
    if (spawnError) child.emit('error', new Error('spawn failed'));
    else child.stdout.write('{"type":"done"}\n');
    child.emit('close', 1);
    await consumer.settled();
    assert.deepEqual(failures, [spawnError ? 'spawn failed' : 'handler failed']);
  }
});
