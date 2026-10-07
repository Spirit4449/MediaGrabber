const fs = require('node:fs/promises');
const path = require('node:path');
const { randomUUID } = require('node:crypto');

function createSharedDownloads({ directory, baseUrl, retentionHours = 72 }) {
  if (!Number.isFinite(retentionHours) || retentionHours <= 0) throw new Error('DOWNLOAD_RETENTION_HOURS must be positive');
  let url;
  if (baseUrl) {
    url = new URL(baseUrl);
    if (!['http:', 'https:'].includes(url.protocol) || url.search || url.hash || url.username || url.password) {
      throw new Error('PUBLIC_DOWNLOAD_BASE_URL must be an HTTP(S) URL without credentials, query or fragment');
    }
  }
  const root = path.resolve(directory);
  const retentionMs = retentionHours * 3600000;
  // Only files created by this module are eligible for cleanup.
  const publishedName = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}--/;
  async function prepare() {
    await fs.mkdir(root, { recursive: true, mode: 0o755 });
  }
  async function publish(source) {
    if (!url) throw new Error('Set PUBLIC_DOWNLOAD_BASE_URL on production to deliver oversized files. The original has been kept.');
    await prepare();
    const filename = path.basename(source).replace(/[^a-zA-Z0-9._-]/g, '_').slice(-160) || 'download';
    const name = `${randomUUID()}--${filename}`;
    const destination = path.join(root, name);
    const staging = path.join(root, `.${name}.partial`);
    try {
      await fs.copyFile(source, staging, fs.constants.COPYFILE_EXCL);
      await fs.chmod(staging, 0o644);
      const now = new Date();
      await fs.utimes(staging, now, now);
      await fs.rename(staging, destination);
    } catch (error) {
      await fs.rm(staging, { force: true }).catch(() => {});
      throw error;
    }
    return { path: destination, url: `${url.href.replace(/\/$/, '')}/${encodeURIComponent(name)}`, expiresAt: new Date(Date.now() + retentionMs) };
  }
  async function cleanup(now = Date.now()) {
    await prepare();
    for (const entry of await fs.readdir(root, { withFileTypes: true })) {
      const stalePartial = entry.name.startsWith('.') && entry.name.endsWith('.partial');
      if (!entry.isFile() || (!publishedName.test(entry.name) && !stalePartial)) continue;
      const file = path.join(root, entry.name);
      const stat = await fs.stat(file).catch(() => null);
      if (stat && now - stat.mtimeMs >= (stalePartial ? 86400000 : retentionMs)) await fs.rm(file, { force: true });
    }
  }
  return { publish, cleanup, retentionHours, root };
}
module.exports = { createSharedDownloads };
