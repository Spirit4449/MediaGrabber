// server.js
require('dotenv').config();
const express = require('express');
const rateLimit = require('express-rate-limit');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { Telegraf } = require('telegraf');

const {
  PORT = 4000,
  BIND_HOST = '127.0.0.1',
  SHARED_SECRET,
  BOT_TOKEN,
  PYTHON_BIN = 'python'
} = process.env;

if (!SHARED_SECRET) throw new Error('SHARED_SECRET missing in env');
if (!BOT_TOKEN) throw new Error('BOT_TOKEN missing in env');

const app = express();
app.use(express.json({ limit: '256kb' }));

// ====== Helpers (shared) ======
function isAllowedTelegramLink(link) {
  return /^https?:\/\/t\.me\/(?:c\/\d+\/\d+|[A-Za-z0-9_]+\/\d+)$/.test(link);
}
function isInviteLink(text) {
  return /t\.me\/(?:\+|joinchat\/)[A-Za-z0-9_-]+$/.test(text.trim()) || /^[A-Za-z0-9_-]{16,}$/.test(text.trim());
}
function spawnDownloader(args) {
  return spawn(process.env.PYTHON_BIN || PYTHON_BIN, [
    path.join(process.cwd(), 'downloader.py'),
    ...args
  ], { cwd: process.cwd() });
}
async function safeUploadAndDelete(telegram, chatId, filePath, { caption, forceVideo } = {}) {
  const ext = (path.extname(filePath) || '').toLowerCase();
  const isVideo = forceVideo || ['.mp4', '.mov', '.mkv', '.webm'].includes(ext);
  const stream = fs.createReadStream(filePath);
  if (isVideo) {
    await telegram.sendVideo(chatId, { source: stream }, { caption: caption || '' });
  } else {
    await telegram.sendDocument(chatId, { source: stream }, { caption: caption || '' });
  }
  try { await fs.promises.rm(filePath, { force: true }); } catch {}
}

// ====== Secure API endpoint (unchanged) ======
const limiter = rateLimit({ windowMs: 60_000, max: 30 });
app.use('/api/download', limiter);

function verifyHmac(req, res, next) {
  const sig = req.get('x-signature') || '';
  const body = JSON.stringify(req.body || {});
  const mac = crypto.createHmac('sha256', SHARED_SECRET).update(body).digest('hex');
  try {
    if (crypto.timingSafeEqual(Buffer.from(mac), Buffer.from(sig))) return next();
  } catch {}
  return res.status(401).json({ error: 'Invalid signature' });
}

app.post('/api/download', verifyHmac, async (req, res) => {
  try {
    const { link, chat_id, caption, forceVideo } = req.body || {};
    if (!link || !chat_id) return res.status(400).json({ error: 'link and chat_id required' });
    if (!isAllowedTelegramLink(link)) return res.status(400).json({ error: 'Invalid link format' });

    const DOWNLOAD_ROOT = path.join(process.cwd(), 'downloads');
    await fs.promises.mkdir(DOWNLOAD_ROOT, { recursive: true });
    try { await fs.promises.chmod(DOWNLOAD_ROOT, 0o700); } catch {}

    const bot = new Telegraf(BOT_TOKEN);
    await bot.telegram.sendMessage(chat_id, `🟢 Starting download...\n${link}`);

    const py = spawnDownloader(['--link', link, '--outdir', DOWNLOAD_ROOT]);

    let buffer = ''; let lastPctReported = -10;
    py.stdout.on('data', async chunk => {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let ev; try { ev = JSON.parse(line); } catch { continue; }

        if (ev.type === 'status' && ev.text) {
          await bot.telegram.sendMessage(chat_id, `ℹ️ ${ev.text}`);
        } else if (ev.type === 'progress' && typeof ev.pct === 'number') {
          if (ev.pct - lastPctReported >= 10 || ev.pct === 100) {
            lastPctReported = ev.pct;
            await bot.telegram.sendChatAction(chat_id, 'upload_document');
            await bot.telegram.sendMessage(chat_id, `📥 ${ev.pct}% ${ev.text ? `(${ev.text})` : ''}`);
          }
        } else if (ev.type === 'error') {
          await bot.telegram.sendMessage(chat_id, `❌ ${ev.text || 'Error'}`);
        } else if (ev.type === 'done' && ev.path) {
          try {
            await safeUploadAndDelete(bot.telegram, chat_id, ev.path, {
              caption: caption || '✅ Download complete',
              forceVideo: !!forceVideo
            });
            await bot.telegram.sendMessage(chat_id, `✅ Uploaded to chat and removed from server`);
          } catch (e) {
            await bot.telegram.sendMessage(chat_id, `❌ Upload failed: ${e.message}`);
          }
        }
      }
    });

    let stderrText = '';
    py.stderr.on('data', chunk => { stderrText += chunk.toString('utf8'); });
    py.on('close', async (code) => {
      if (stderrText) await bot.telegram.sendMessage(chat_id, `🪵 Log: ${stderrText.slice(0, 1500)}`);
      if (code === 0) return res.json({ ok: true });
      return res.status(500).json({ ok: false, code, stderr: stderrText });
    });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

// ====== Telegram Bot (polling) ======
const bot = new Telegraf(BOT_TOKEN);

// simple in-memory state per chat
const state = new Map(); // chatId -> { awaitingInviteFor: <link> }

bot.start(async (ctx) => {
  await ctx.reply(
    '👋 Send me a Telegram post link (e.g. https://t.me/c/123456789/42). I will fetch the media and upload it here.\n' +
    'If it’s a private link and I’m not in the channel, I’ll ask you for an invite link.'
  );
});

bot.on('text', async (ctx) => {
  const chatId = ctx.chat.id;
  const text = (ctx.message.text || '').trim();

  // If we are awaiting an invite
  const st = state.get(chatId);
  if (st && st.awaitingInviteFor) {
    if (!isInviteLink(text)) {
      return ctx.reply('🔑 Please send a valid invite link (t.me/+XXXX or t.me/joinchat/XXXX).');
    }
    const invite = text;
    const link = st.awaitingInviteFor;
    state.delete(chatId);
    await handleDownloadFlow(ctx, link, { invite });
    return;
  }

  // Otherwise expect a post link
  if (!isAllowedTelegramLink(text)) {
    return ctx.reply('🔗 Please send a valid Telegram post link (public or private).');
  }

  await handleDownloadFlow(ctx, text);
});

async function handleDownloadFlow(ctx, link, opts = {}) {
  const chatId = ctx.chat.id;
  const DOWNLOAD_ROOT = path.join(process.cwd(), 'downloads');
  await fs.promises.mkdir(DOWNLOAD_ROOT, { recursive: true });
  try { await fs.promises.chmod(DOWNLOAD_ROOT, 0o700); } catch {}

  await ctx.reply(`🟢 Processing link:\n${link}`);

  // 1) Preflight: check access
  const preArgs = ['--link', link, '--outdir', DOWNLOAD_ROOT, '--preflight'];
  if (opts.invite) preArgs.push('--invite', opts.invite);
  let needInvite = false; let expected = null; let hasMedia = null;
  await new Promise((resolve) => {
    const py = spawnDownloader(preArgs);
    let buffer = '';
    py.stdout.on('data', chunk => {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        try {
          const ev = JSON.parse(line);
          if (ev.type === 'status' && ev.text) ctx.reply(`ℹ️ ${ev.text}`).catch(()=>{});
          if (ev.type === 'need_invite') needInvite = true;
          if (ev.type === 'ok') { expected = ev.expected; hasMedia = ev.has_media; }
          if (ev.type === 'error') ctx.reply(`❌ ${ev.text || 'Error'}`).catch(()=>{});
        } catch {}
      }
    });
    py.on('close', () => resolve());
  });

  if (needInvite && !opts.invite) {
    state.set(chatId, { awaitingInviteFor: link });
    return ctx.reply('🔐 I need an invite link to join that channel/group.\nPlease send the invite (t.me/+XXXX or t.me/joinchat/XXXX).');
  }

  if (hasMedia === false) {
    return ctx.reply('⚠️ That post has no media to download.');
  }

  // 2) Do the download
  await ctx.reply('🚀 Starting download...');
  const args = ['--link', link, '--outdir', DOWNLOAD_ROOT];
  if (opts.invite) args.push('--invite', opts.invite);

  let lastPctReported = -10;
  await new Promise((resolve) => {
    const py = spawnDownloader(args);
    let buffer = '';
    py.stdout.on('data', async chunk => {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let ev; try { ev = JSON.parse(line); } catch { continue; }

        if (ev.type === 'status' && ev.text) {
          ctx.reply(`ℹ️ ${ev.text}`).catch(()=>{});
        } else if (ev.type === 'progress' && typeof ev.pct === 'number') {
          if (ev.pct - lastPctReported >= 10 || ev.pct === 100) {
            lastPctReported = ev.pct;
            await ctx.telegram.sendChatAction(chatId, 'upload_document').catch(()=>{});
            ctx.reply(`📥 ${ev.pct}% ${ev.text ? `(${ev.text})` : ''}`).catch(()=>{});
          }
        } else if (ev.type === 'error') {
          ctx.reply(`❌ ${ev.text || 'Error'}`).catch(()=>{});
        } else if (ev.type === 'done' && ev.path) {
          (async () => {
            try {
              await safeUploadAndDelete(ctx.telegram, chatId, ev.path, { caption: '✅ Download complete' });
              await ctx.reply('✅ Uploaded to chat and removed from server');
            } catch (e) {
              await ctx.reply(`❌ Upload failed: ${e.message}`);
            }
          })().catch(()=>{});
        }
      }
    });
    py.on('close', () => resolve());
  });
}

// Health check
app.get('/healthz', (_req, res) => res.json({ ok: true }));

// Start HTTP + Bot (polling)
app.listen(Number(PORT), BIND_HOST, () => {
  console.log(`Server listening on http://${BIND_HOST}:${PORT}`);
});
bot.launch().then(() => console.log('Bot polling started')).catch(console.error);

// Graceful stop (PM2 / Docker)
process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));
