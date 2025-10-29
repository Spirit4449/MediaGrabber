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
const bot = new Telegraf(BOT_TOKEN);

// ---------- settings ----------
const INVITE_WAIT_MS = 5 * 60 * 1000; // 5 minutes to send invite before expiring

// ---------- helpers ----------
function isAllowedTelegramLink(link) {
  return /^https?:\/\/t\.me\/(?:c\/\d+\/\d+|[A-Za-z0-9_]+\/\d+)$/.test(link);
}
function isInviteLink(text) {
  return /t\.me\/(?:\+|joinchat\/)[A-Za-z0-9_-]+$/.test(text.trim()) || /^[A-Za-z0-9_-]{16,}$/.test(text.trim());
}
function spawnDownloader(args) {
  console.log('[spawn] python downloader.py', args.join(' '));
  return spawn(process.env.PYTHON_BIN || PYTHON_BIN, [
    path.join(process.cwd(), 'downloader.py'),
    ...args
  ], { cwd: process.cwd() });
}
async function safeUploadAndDelete(telegram, chatId, filePath, { caption, forceVideo } = {}) {
  const base = path.basename(filePath);
  console.log('[upload] sending', base, 'to chat', chatId);

  const stream = fs.createReadStream(filePath);
  const inputFile = { source: stream, filename: base };

  const ext = (path.extname(base) || '').toLowerCase();
  const isVideo = forceVideo || ['.mp4', '.mov', '.mkv', '.webm'].includes(ext);
  const isPhoto = ['.jpg', '.jpeg', '.png', '.gif', '.webp'].includes(ext);

  if (isVideo) {
    await telegram.sendVideo(chatId, inputFile, { caption: caption || '' });
  } else if (isPhoto) {
    await telegram.sendPhoto(chatId, inputFile, { caption: caption || '' });
  } else {
    await telegram.sendDocument(chatId, inputFile, { caption: caption || '' });
  }
  try { await fs.promises.rm(filePath, { force: true }); } catch {}
}
function fmtMB(bytes) {
  if (!bytes && bytes !== 0) return '??';
  return (bytes / (1024 * 1024)).toFixed(1);
}
function progressBar(pct) {
  const total = 20;
  const filled = Math.max(0, Math.min(total, Math.round((pct / 100) * total)));
  return '█'.repeat(filled) + '░'.repeat(total - filled);
}
function progressText(downloaded, total, pct) {
  const d = fmtMB(downloaded), t = fmtMB(total);
  const bar = progressBar(Math.max(0, Math.min(100, pct || 0)));
  return `📥 ${Math.round(pct || 0)}% [${bar}] ${d} MB / ${t} MB`;
}

// ---------- bot state ----------
/**
 * state: Map<chatId, { awaitingInviteFor: string, expiresAt: number }>
 */
const state = new Map();

function inviteWaitActive(st) {
  return !!st && Date.now() < (st.expiresAt || 0);
}
function clearInviteWait(chatId) {
  state.delete(chatId);
  console.log('[invite] cleared for chat', chatId);
}

// ---------- bot commands ----------
bot.start(async (ctx) => {
  await ctx.reply('👋 Send a Telegram post link. I will fetch the media and upload it here.\nIf it’s private and I’m not a member, I’ll ask for an invite link.\nType /stop or "stop" to cancel when asked for an invite.');
});

bot.command(['stop', 'cancel'], async (ctx) => {
  const chatId = ctx.chat.id;
  const st = state.get(chatId);
  if (inviteWaitActive(st)) {
    clearInviteWait(chatId);
    await ctx.reply('🛑 Canceled. You can now send a new Telegram post link anytime.');
  } else {
    await ctx.reply('Nothing to cancel. Send me a Telegram post link to begin.');
  }
});

// ---------- bot text handler ----------
bot.on('text', async (ctx) => {
  const chatId = ctx.chat.id;
  const text = (ctx.message.text || '').trim();

  // allow plain "stop"/"cancel" during invite wait
  if (/^(stop|cancel)$/i.test(text)) {
    const st = state.get(chatId);
    if (inviteWaitActive(st)) {
      clearInviteWait(chatId);
      return ctx.reply('🛑 Canceled. You can now send a new Telegram post link.');
    }
  }

  const st = state.get(chatId);
  // If awaiting invite, handle invite or allow stop
  if (inviteWaitActive(st)) {
    if (!isInviteLink(text)) {
      return ctx.reply('🔑 Please send a valid invite link (e.g., `https://t.me/+INVITEHASH`) or type `stop` to cancel.', { parse_mode: 'Markdown' });
    }
    const invite = text;
    const link = st.awaitingInviteFor;
    clearInviteWait(chatId);
    await handleDownloadFlow(ctx, link, { invite });
    return;
  } else {
    // If expired, clear and continue
    if (st) clearInviteWait(chatId);
  }

  // Otherwise expect a post link
  if (!isAllowedTelegramLink(text)) {
    return ctx.reply('🔗 Please send a valid Telegram post link.');
  }
  await handleDownloadFlow(ctx, text);
});

// ---------- main bot flow ----------
async function handleDownloadFlow(ctx, link, opts = {}) {
  const chatId = ctx.chat.id;
  const DOWNLOAD_ROOT = path.join(process.cwd(), 'downloads');
  await fs.promises.mkdir(DOWNLOAD_ROOT, { recursive: true });

  console.log('[bot] preflight for', link);
  await ctx.reply(`🟢 Received link:\n${link}`);

  // Preflight
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
        let ev; try { ev = JSON.parse(line); } catch { console.log('[preflight log]', line); continue; }
        if (ev.type === 'need_invite') needInvite = true;
        if (ev.type === 'ok') { expected = ev.expected; hasMedia = ev.has_media; }
      }
    });
    py.on('close', () => resolve());
  });

  if (needInvite && !opts.invite) {
    // set wait state with expiry
    state.set(chatId, { awaitingInviteFor: link, expiresAt: Date.now() + INVITE_WAIT_MS });
    await ctx.reply(
      '🔐 I need an invite link to join that channel/group.\n' +
      'Please send: `https://t.me/+INVITEHASH`\n' +
      'Type `stop` to cancel. (This request auto-expires in 5 minutes.)',
      { parse_mode: 'Markdown' }
    );
    return;
  }

  if (hasMedia === false) {
    await ctx.reply('⚠️ That post has no media to download.');
    return;
  }

  const preTxt = expected ? `Expected: ${fmtMB(expected)} MB` : 'Starting…';
  const m = await ctx.reply(`📥 0% [░░░░░░░░░░░░░░░░░░] 0.0 MB / ${expected ? fmtMB(expected) : '??'} MB\n${preTxt}`);
  const progressMsgId = m.message_id;
  console.log('[bot] starting download for', link);

  startBackgroundDownload(ctx, link, progressMsgId, { invite: opts.invite });
}

function startBackgroundDownload(ctx, link, progressMsgId, opts = {}) {
  const chatId = ctx.chat.id;
  const DOWNLOAD_ROOT = path.join(process.cwd(), 'downloads');
  const args = ['--link', link, '--outdir', DOWNLOAD_ROOT];
  if (opts.invite) args.push('--invite', opts.invite);

  const py = spawnDownloader(args);
  let buffer = '';
  let lastPctLogged = -10;

  py.stdout.on('data', async chunk => {
    buffer += chunk.toString('utf8');
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;

      let ev; try { ev = JSON.parse(line); } catch { console.log('[dl log]', line); continue; }

      if (ev.type === 'progress') {
        const pct = typeof ev.pct === 'number' ? ev.pct : 0;
        const txt = progressText(ev.downloaded, ev.total, pct);
        try {
          await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, txt);
        } catch {}
        if (pct - lastPctLogged >= 10 || pct === 100) {
          lastPctLogged = pct;
          console.log(`[progress] ${pct}% ${fmtMB(ev.downloaded)}MB/${fmtMB(ev.total)}MB`);
        }
      } else if (ev.type === 'error') {
        console.error('[error]', ev.code || '', ev.text || '');
        await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, `❌ ${ev.text || 'Failed'}`).catch(()=>{});
      } else if (ev.type === 'done' && ev.path) {
        console.log('[done] path:', ev.path, 'size:', ev.size);
        try {
          await safeUploadAndDelete(ctx.telegram, chatId, ev.path, { caption: '' });
          await ctx.telegram.deleteMessage(chatId, progressMsgId).catch(()=>{}); // success: delete progress
        } catch (e) {
          console.error('[upload fail]', e.message);
          await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, `❌ Upload failed: ${e.message}`).catch(()=>{});
        }
      }
    }
  });

  py.stderr.on('data', chunk => console.error('[downloader stderr]', chunk.toString('utf8')));
  py.on('close', code => console.log('[process close]', code));
}

// ---------- express (optional secure API) ----------
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

    console.log('[api] request', { link, chat_id });

    const DOWNLOAD_ROOT = path.join(process.cwd(), 'downloads');
    await fs.promises.mkdir(DOWNLOAD_ROOT, { recursive: true });
    try { await fs.promises.chmod(DOWNLOAD_ROOT, 0o700); } catch {}

    const pmsg = await bot.telegram.sendMessage(chat_id, '📥 0% [░░░░░░░░░░░░░░░░░░] 0.0 MB / ?? MB');
    let progressMsgId = pmsg.message_id;

    const py = spawnDownloader(['--link', link, '--outdir', DOWNLOAD_ROOT]);

    let buffer = '';
    let lastPctLogged = -10;

    py.stdout.on('data', async chunk => {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;

        let ev; try { ev = JSON.parse(line); } catch { console.log('[downloader log]', line); continue; }

        if (ev.type === 'progress') {
          const pct = typeof ev.pct === 'number' ? ev.pct : 0;
          const txt = progressText(ev.downloaded, ev.total, pct);
          try {
            await bot.telegram.editMessageText(chat_id, progressMsgId, undefined, txt);
          } catch (e) {
            console.warn('[edit fail]', e.message);
            const np = await bot.telegram.sendMessage(chat_id, txt);
            progressMsgId = np.message_id;
          }
          if (pct - lastPctLogged >= 10 || pct === 100) {
            lastPctLogged = pct;
            console.log(`[progress] ${pct}% ${fmtMB(ev.downloaded)}MB/${fmtMB(ev.total)}MB`);
          }
        } else if (ev.type === 'error') {
          console.error('[error]', ev.code || '', ev.text || '');
          await bot.telegram.editMessageText(chat_id, progressMsgId, undefined, `❌ ${ev.text || 'Failed'}`).catch(()=>{});
        } else if (ev.type === 'done' && ev.path) {
          console.log('[done] path:', ev.path, 'size:', ev.size);
          try {
            await safeUploadAndDelete(bot.telegram, chat_id, ev.path, { caption: '', forceVideo: !!forceVideo });
            await bot.telegram.deleteMessage(chat_id, progressMsgId).catch(()=>{});
          } catch (e) {
            console.error('[upload fail]', e.message);
            await bot.telegram.editMessageText(chat_id, progressMsgId, undefined, `❌ Upload failed: ${e.message}`).catch(()=>{});
          }
        }
      }
    });

    py.stderr.on('data', chunk => {
      console.error('[downloader stderr]', chunk.toString('utf8'));
    });

    py.on('close', (code) => {
      console.log('[process close] code:', code);
    });

    res.json({ ok: true });
  } catch (e) {
    console.error('[api error]', e);
    return res.status(500).json({ error: e.message });
  }
});

// ---------- health & start ----------
app.get('/healthz', (_req, res) => res.json({ ok: true }));

app.listen(Number(PORT), BIND_HOST, () => {
  console.log(`Server listening on http://${BIND_HOST}:${PORT}`);
});
bot.launch().then(() => console.log('Bot polling started')).catch(console.error);
process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));
