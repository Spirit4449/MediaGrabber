// stdout listeners must parse synchronously, then handle events in order.
// Otherwise an awaited progress edit can overwrite a later delivery/error status.
function consumeDownloader(child, { onEvent, onFailure, onLog = console.log }) {
  let buffer = '', terminal = false;
  let queue = Promise.resolve();
  function enqueue(line) {
    if (!line.trim()) return;
    let event;
    try { event = JSON.parse(line); }
    catch { onLog(line); return; }
    queue = queue.then(async () => {
      if (terminal) return;
      if (['done', 'error', 'need_invite'].includes(event.type)) terminal = true;
      await onEvent(event);
    }).catch(async error => {
      terminal = true;
      await onFailure(error).catch(() => {});
    });
  }
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', chunk => {
    buffer += chunk;
    let index;
    while ((index = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      enqueue(line);
    }
  });
  child.on('error', error => {
    queue = queue.then(async () => { terminal = true; await onFailure(error); }).catch(() => {});
  });
  child.on('close', (code, signal) => {
    if (buffer.trim()) enqueue(buffer);
    buffer = '';
    queue = queue.then(async () => {
      onLog(`[process close] ${code}${signal ? ` (${signal})` : ''}`);
      if (!terminal) {
        terminal = true;
        await onFailure(new Error(`Download worker exited without a completion event (code ${code}${signal ? `, ${signal}` : ''}). Check the worker logs.`));
      }
    }).catch(() => {});
  });
  return { settled: () => queue };
}
module.exports = { consumeDownloader };
