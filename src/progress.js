function fmtMB(bytes) {
  if (!bytes && bytes !== 0) return "??";
  return (bytes / (1024 * 1024)).toFixed(1);
}
function progressBar(pct) {
  const total = 20;
  const filled = Math.max(0, Math.min(total, Math.round((pct / 100) * total)));
  return "█".repeat(filled) + "░".repeat(total - filled);
}
function progressText(downloaded, total, pct) {
  const d = fmtMB(downloaded),
    t = fmtMB(total);
  const bar = progressBar(Math.max(0, Math.min(100, pct || 0)));
  return `📥 ${Math.round(pct || 0)}% [${bar}] ${d} MB / ${t} MB`;
}

module.exports = { fmtMB, progressText };
