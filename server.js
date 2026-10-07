const path = require("node:path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const express = require("express");
const { createSharedDownloads } = require("./src/shared-downloads");
const { createCompressor } = require("./src/media-compression");
const { Telegraf } = require("telegraf");
const { createMediaDelivery } = require("./src/media-delivery");
const { createDownloadFlow } = require("./src/download-flow");
const { registerBot } = require("./src/bot");
const { registerApi } = require("./src/api");

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
  directory: process.env.SHARED_DOWNLOAD_DIR || path.join(__dirname, "shared-downloads"),
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

const safeUploadAndDelete = createMediaDelivery({
  maxUploadMb: MAX_BOT_UPLOAD_MB,
  sendCompressedCopy: SEND_COMPRESSED_COPY,
  sharedDownloads,
  compressor,
});
const flow = createDownloadFlow({ rootDir: __dirname, pythonBin: PYTHON_BIN, safeUploadAndDelete });
registerBot(bot, flow);
registerApi(app, { bot, sharedSecret: SHARED_SECRET, rootDir: __dirname, startBackgroundDownload: flow.startBackgroundDownload });

app.listen(Number(PORT), BIND_HOST, () => {
  console.log(`Server listening on http://${BIND_HOST}:${PORT}`);
});
bot
  .launch()
  .then(() => console.log("Bot polling started"))
  .catch(console.error);
process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
