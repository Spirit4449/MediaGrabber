const fs = require('node:fs');
const path = require('node:path');

// Successful empty long-poll responses count as healthy, even on quiet days.
function trackPolling(telegram, report) {
  const callApi = telegram.callApi.bind(telegram);
  telegram.callApi = async (method, ...args) => {
    const result = await callApi(method, ...args);
    if (method === 'getUpdates') report();
    return result;
  };
}

function writeHeartbeat(file) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify({ pid: process.pid, polled_at: Date.now() / 1000 }));
  fs.renameSync(temporary, file);
}

function launchBot(bot, shutdown, log = console.log) {
  return bot.launch({}, () => log('Bot connected; starting polling'))
    .then(() => shutdown('Polling stopped', 1))
    .catch(error => {
      // Avoid logging request objects containing the bot token.
      console.error('[bot fatal]', error.message);
      shutdown('Polling failed', 1);
    });
}

module.exports = { trackPolling, writeHeartbeat, launchBot };
