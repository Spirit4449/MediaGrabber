require("dotenv").config();
const express = require("express");
const rateLimit = require("express-rate-limit");
const crypto = require("crypto");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { createSharedDownloads } = require("./shared-downloads");
const { createCompressor } = require("./media-compression");
const { consumeDownloader } = require("./downloader-events");
const { Telegraf } = require("telegraf");

const {
  PORT = 4000,
  BIND_HOST = "127.0.0.1",
  SHARED_SECRET,
  BOT_TOKEN,
  PYTHON_BIN = "python",
} = process.env;

if (!SHARED_SECRET) throw new Error("SHARED_SECRET missing in env");
if (!BOT_TOKEN) throw new Error("BOT_TOKEN missing in env");

const app = express();
app.use(express.json({ limit: "256kb" }));
const bot = new Telegraf(BOT_TOKEN);
const MAX_BOT_UPLOAD_MB = Number(process.env.BOT_MAX_UPLOAD_MB || 49);
const COMPRESS_TARGET_MB = Number(
  process.env.BOT_COMPRESS_TARGET_MB || Math.max(1, MAX_BOT_UPLOAD_MB - 2),
);
const COMPRESS_AUDIO_KBPS = Number(process.env.BOT_COMPRESS_AUDIO_KBPS || 192);
const COMPRESS_MIN_VIDEO_KBPS = Number(
  process.env.BOT_COMPRESS_MIN_VIDEO_KBPS || 300,
);
const COMPRESS_MAX_VIDEO_KBPS = Number(
  process.env.BOT_COMPRESS_MAX_VIDEO_KBPS || 2500,
);
const FFMPEG_PATH = process.env.FFMPEG_PATH || "ffmpeg";
const FFPROBE_PATH = process.env.FFPROBE_PATH || "ffprobe";

if (!Number.isFinite(MAX_BOT_UPLOAD_MB) || MAX_BOT_UPLOAD_MB <= 0) throw new Error("BOT_MAX_UPLOAD_MB must be positive");
const SEND_COMPRESSED_COPY = process.env.BOT_SEND_COMPRESSED_COPY === "true";
const sharedDownloads = createSharedDownloads({
  directory: process.env.SHARED_DOWNLOAD_DIR || path.join(process.cwd(), "shared-downloads"),
  baseUrl: process.env.PUBLIC_DOWNLOAD_BASE_URL,
  retentionHours: Number(process.env.DOWNLOAD_RETENTION_HOURS || 72),
});
const compressor = createCompressor({
  ffmpeg: FFMPEG_PATH, ffprobe: FFPROBE_PATH,
  targetMb: COMPRESS_TARGET_MB, audioKbps: COMPRESS_AUDIO_KBPS,
  minVideoKbps: COMPRESS_MIN_VIDEO_KBPS, maxVideoKbps: COMPRESS_MAX_VIDEO_KBPS,
  preset: process.env.BOT_COMPRESS_PRESET || "medium",
});
if (!process.env.PUBLIC_DOWNLOAD_BASE_URL) console.warn("[downloads] PUBLIC_DOWNLOAD_BASE_URL is missing; oversized originals will be kept but cannot be shared.");
sharedDownloads.cleanup().catch(error => console.error("[downloads cleanup]", error.message));
setInterval(() => sharedDownloads.cleanup().catch(error => console.error("[downloads cleanup]", error.message)), 15 * 60 * 1000).unref();

// ---------- settings ----------
const INVITE_WAIT_MS = 5 * 60 * 1000; // 5 minutes to send invite before expiring

// ---------- helpers ----------
function isAllowedTelegramLink(link) {
  return /^https?:\/\/t\.me\/(?:c\/\d+\/\d+|[A-Za-z0-9_]+\/\d+)$/.test(link);
}
function isInviteLink(text) {
  return (
    /t\.me\/(?:\+|joinchat\/)[A-Za-z0-9_-]+$/.test(text.trim()) ||
    /^[A-Za-z0-9_-]{16,}$/.test(text.trim())
  );
}
function spawnDownloader(args) {
  console.log("[spawn] python downloader.py", args.join(" "));
  return spawn(
    process.env.PYTHON_BIN || PYTHON_BIN,
    [path.join(process.cwd(), "downloader.py"), ...args],
    { cwd: process.cwd() },
  );
}
const VIDEO_EXTS = new Set([
  ".mp4",
  ".mov",
  ".mkv",
  ".webm",
  ".avi",
  ".m4v",
  ".flv",
  ".wmv",
]);
const PHOTO_EXTS = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp"]);
const AUDIO_EXTS = new Set([
  ".mp3",
  ".m4a",
  ".ogg",
  ".oga",
  ".wav",
  ".flac",
  ".aac",
  ".opus",
  ".m4b",
]);
async function safeUploadAndDelete(telegram, chatId, filePath, { caption, forceVideo } = {}) {
  const originalBase = path.basename(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const isVideo = !!forceVideo || VIDEO_EXTS.has(ext);
  const isAudio = AUDIO_EXTS.has(ext);
  const maxBytes = MAX_BOT_UPLOAD_MB * 1024 * 1024;
  const stat = await fs.promises.stat(filePath);
  let originalShared = false;
  let compressed;
  let currentPath = filePath;
  let base = originalBase;
  if (stat.size > maxBytes) {
    console.log("[delivery] publishing original", originalBase);
    const shared = await sharedDownloads.publish(filePath);
    console.log("[delivery] original ready", originalBase);
    // Keep the working original if Telegram rejects the link message, for retry.
    await telegram.sendMessage(chatId,
      `${caption ? caption + "\n" : ""}📥 ${originalBase} · ${fmtMB(stat.size)} MB\nOriginal quality. Available for ${sharedDownloads.retentionHours} hours.`,
      { reply_markup: { inline_keyboard: [[{ text: "Download original", url: shared.url }]] } });
    originalShared = true;
    if (!SEND_COMPRESSED_COPY || (!isVideo && !isAudio)) {
      await fs.promises.rm(filePath, { force: true });
      return;
    }
  }
  if (originalShared || (isAudio && ext !== ".mp3")) {
    compressed = await compressor.compress(filePath, { isVideo, isAudio, maxBytes });
    if (!compressed.path) {
      if (!originalShared) throw new Error(`Audio conversion failed: ${compressed.error}. Original kept.`);
      await telegram.sendMessage(chatId, `Original download is ready. Compressed copy unavailable: ${compressed.error}`);
      await fs.promises.rm(filePath, { force: true });
      return;
    }
    currentPath = compressed.path;
    base = path.basename(originalBase, ext) + compressed.ext;
  }
  const stream = fs.createReadStream(currentPath);
  const inputFile = { source: stream, filename: base };
  try {
    const options = { caption: originalShared ? "Compressed copy — original available above" : (caption || "") };
    if (isVideo) {
      if (compressed) Object.assign(options, { width: compressed.width, height: compressed.height, duration: Math.ceil(compressed.duration), supports_streaming: true });
      await telegram.sendVideo(chatId, inputFile, options);
    } else if (PHOTO_EXTS.has(ext)) {
      await telegram.sendPhoto(chatId, inputFile, options);
    } else {
      await telegram.sendDocument(chatId, inputFile, options);
    }
    await fs.promises.rm(filePath, { force: true });
  } catch (error) {
    if (!originalShared) throw error;
    console.error("[preview upload failed]", error.message);
    await fs.promises.rm(filePath, { force: true });
    // The original link was already delivered successfully.
  } finally {
    stream.destroy();
    if (compressed) await compressed.cleanup();
  }
}
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
  console.log("[invite] cleared for chat", chatId);
}

// ---------- bot commands ----------
bot.start(async (ctx) => {
  await ctx.reply(
    '👋 Send a Telegram post link. I will fetch the media and upload it here.\nIf it’s private and I’m not a member, I’ll ask for an invite link.\nType /stop or "stop" to cancel when asked for an invite.',
  );
});

bot.command(["stop", "cancel"], async (ctx) => {
  const chatId = ctx.chat.id;
  const st = state.get(chatId);
  if (inviteWaitActive(st)) {
    clearInviteWait(chatId);
    await ctx.reply(
      "🛑 Canceled. You can now send a new Telegram post link anytime.",
    );
  } else {
    await ctx.reply(
      "Nothing to cancel. Send me a Telegram post link to begin.",
    );
  }
});

// ---------- bot text handler ----------
bot.on("text", async (ctx) => {
  const chatId = ctx.chat.id;
  const text = (ctx.message.text || "").trim();

  // allow plain "stop"/"cancel" during invite wait
  if (/^(stop|cancel)$/i.test(text)) {
    const st = state.get(chatId);
    if (inviteWaitActive(st)) {
      clearInviteWait(chatId);
      return ctx.reply(
        "🛑 Canceled. You can now send a new Telegram post link.",
      );
    }
  }

  const st = state.get(chatId);
  // If awaiting invite, handle invite or allow stop
  if (inviteWaitActive(st)) {
    if (!isInviteLink(text)) {
      return ctx.reply(
        "🔑 Please send a valid invite link (e.g., `https://t.me/+INVITEHASH`) or type `stop` to cancel.",
        { parse_mode: "Markdown" },
      );
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
    return ctx.reply("🔗 Please send a valid Telegram post link.");
  }
  await handleDownloadFlow(ctx, text);
});

// ---------- main bot flow ----------
async function handleDownloadFlow(ctx, link, opts = {}) {
  const chatId = ctx.chat.id;
  const DOWNLOAD_ROOT = path.join(process.cwd(), "downloads");
  await fs.promises.mkdir(DOWNLOAD_ROOT, { recursive: true });

  console.log("[bot] preflight for", link);
  await ctx.reply(`🟢 Received link:\n${link}`);

  // Preflight
  const preArgs = ["--link", link, "--outdir", DOWNLOAD_ROOT, "--preflight"];
  if (opts.invite) preArgs.push("--invite", opts.invite);

  let needInvite = false;
  let expected = null;
  let hasMedia = null;

  await new Promise((resolve) => {
    const py = spawnDownloader(preArgs);
    let buffer = "";
    py.stdout.on("data", (chunk) => {
      buffer += chunk.toString("utf8");
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let ev;
        try {
          ev = JSON.parse(line);
        } catch {
          console.log("[preflight log]", line);
          continue;
        }
        if (ev.type === "need_invite") needInvite = true;
        if (ev.type === "ok") {
          expected = ev.expected;
          hasMedia = ev.has_media;
        }
      }
    });
    py.on("close", () => resolve());
  });

  if (needInvite && !opts.invite) {
    // set wait state with expiry
    state.set(chatId, {
      awaitingInviteFor: link,
      expiresAt: Date.now() + INVITE_WAIT_MS,
    });
    await ctx.reply(
      "🔐 I need an invite link to join that channel/group.\n" +
        "Please send: `https://t.me/+INVITEHASH`\n" +
        "Type `stop` to cancel. (This request auto-expires in 5 minutes.)",
      { parse_mode: "Markdown" },
    );
    return;
  }

  if (hasMedia === false) {
    await ctx.reply("⚠️ That post has no media to download.");
    return;
  }

  const preTxt = expected ? `Expected: ${fmtMB(expected)} MB` : "Starting…";
  const m = await ctx.reply(
    `📥 0% [░░░░░░░░░░░░░░░░░░] 0.0 MB / ${expected ? fmtMB(expected) : "??"} MB\n${preTxt}`,
  );
  const progressMsgId = m.message_id;
  console.log("[bot] starting download for", link);

  startBackgroundDownload(ctx, link, progressMsgId, { invite: opts.invite });
}

function startBackgroundDownload(ctx, link, progressMsgId, opts = {}) {
  const chatId = ctx.chat.id;
  const args = ["--link", link, "--outdir", path.join(process.cwd(), "downloads", crypto.randomUUID())];
  if (opts.invite) args.push("--invite", opts.invite);
  const py = spawnDownloader(args);
  let lastPctLogged = -10;
  consumeDownloader(py, {
    onFailure: async error => {
      console.error("[worker failure]", error.message);
      await ctx.telegram.editMessageText(chatId, progressMsgId, undefined,
        `❌ ${error.message}`).catch(() => {});
    },
    onEvent: async ev => {
      if (ev.type === "progress") {
        const pct = typeof ev.pct === "number" ? ev.pct : 0;
        const text = pct >= 100 ? "✅ Download complete. Finalizing file…" : progressText(ev.downloaded, ev.total, pct);
        await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, text).catch(() => {});
        if (pct - lastPctLogged >= 10 || pct === 100) {
          lastPctLogged = pct;
          console.log(`[progress] ${pct}% ${fmtMB(ev.downloaded)}MB/${fmtMB(ev.total)}MB`);
        }
      } else if (ev.type === "error") {
        console.error("[error]", ev.code || "", ev.text || "");
        await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, `❌ ${ev.text || "Failed"}`).catch(() => {});
      } else if (ev.type === "done" && ev.path) {
        console.log("[done] path:", ev.path, "size:", ev.size);
        await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, "✅ Download complete. Preparing delivery…").catch(() => {});
        try {
          await safeUploadAndDelete(ctx.telegram, chatId, ev.path, { caption: opts.caption || "", forceVideo: !!opts.forceVideo });
          await ctx.telegram.deleteMessage(chatId, progressMsgId).catch(() => {});
        } catch (error) {
          console.error("[delivery failed]", error.message);
          await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, `❌ Delivery failed: ${error.message}`).catch(() => {});
        }
      } else if (ev.type === "need_invite") {
        await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, "❌ Access changed. Please send the post link again and provide an invite.").catch(() => {});
      }
    },
  });
  py.stderr.on("data", chunk => console.error("[downloader stderr]", chunk.toString("utf8")));
}

// ---------- express (optional secure API) ----------
const limiter = rateLimit({ windowMs: 60_000, max: 30 });
app.use("/api/download", limiter);

function verifyHmac(req, res, next) {
  const sig = req.get("x-signature") || "";
  const body = JSON.stringify(req.body || {});
  const mac = crypto
    .createHmac("sha256", SHARED_SECRET)
    .update(body)
    .digest("hex");
  try {
    if (crypto.timingSafeEqual(Buffer.from(mac), Buffer.from(sig)))
      return next();
  } catch {}
  return res.status(401).json({ error: "Invalid signature" });
}

app.post("/api/download", verifyHmac, async (req, res) => {
  try {
    const { link, chat_id, caption, forceVideo } = req.body || {};
    if (!link || !chat_id)
      return res.status(400).json({ error: "link and chat_id required" });
    if (!isAllowedTelegramLink(link))
      return res.status(400).json({ error: "Invalid link format" });

    console.log("[api] request", { link, chat_id });

    const DOWNLOAD_ROOT = path.join(process.cwd(), "downloads");
    await fs.promises.mkdir(DOWNLOAD_ROOT, { recursive: true });
    try {
      await fs.promises.chmod(DOWNLOAD_ROOT, 0o700);
    } catch {}

    const pmsg = await bot.telegram.sendMessage(
      chat_id,
      "📥 0% [░░░░░░░░░░░░░░░░░░] 0.0 MB / ?? MB",
    );
    startBackgroundDownload({ chat: { id: chat_id }, telegram: bot.telegram }, link,
      pmsg.message_id, { caption: caption || "", forceVideo: !!forceVideo });

    res.json({ ok: true });
  } catch (e) {
    console.error("[api error]", e);
    return res.status(500).json({ error: e.message });
  }
});

// ---------- health & start ----------
app.get("/healthz", (_req, res) => res.json({ ok: true }));

app.listen(Number(PORT), BIND_HOST, () => {
  console.log(`Server listening on http://${BIND_HOST}:${PORT}`);
});
bot
  .launch()
  .then(() => console.log("Bot polling started"))
  .catch(console.error);
process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
