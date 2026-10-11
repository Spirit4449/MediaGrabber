const test = require('node:test');
const assert = require('node:assert/strict');
const { trackPolling, launchBot } = require('../src/bot-health');

test('empty successful polls count as health; unrelated calls and failed polls do not', async () => {
  let reports = 0, fail = false;
  const telegram = { async callApi(method) { if (fail) throw new Error('offline'); return []; } };
  trackPolling(telegram, () => reports++);
  await telegram.callApi('getUpdates');
  await telegram.callApi('getMe');
  fail = true;
  await assert.rejects(telegram.callApi('getUpdates'));
  assert.equal(reports, 1);
});

test('fatal polling rejection requests a failed exit instead of leaving HTTP alive', async () => {
  const exits = [];
  await launchBot({ launch: async () => { throw new Error('409 conflict'); } }, (...args) => exits.push(args));
  assert.deepEqual(exits, [['Polling failed', 1]]);
});

test('unexpected normal polling stop also requests restart', async () => {
  const exits = [];
  await launchBot({ launch: async () => {} }, (...args) => exits.push(args));
  assert.deepEqual(exits, [['Polling stopped', 1]]);
});
