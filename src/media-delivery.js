const fs = require("node:fs");
const path = require("node:path");
const { fmtMB } = require("./progress");

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

function createMediaDelivery({ maxUploadMb, sendCompressedCopy = false, sharedDownloads, compressor }) {
  async function safeUploadAndDelete(telegram, chatId, filePath, { caption, forceVideo } = {}) {
    const originalBase = path.basename(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const isVideo = !!forceVideo || VIDEO_EXTS.has(ext);
    const isAudio = AUDIO_EXTS.has(ext);
    const maxBytes = maxUploadMb * 1024 * 1024;
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
      if (!sendCompressedCopy || (!isVideo && !isAudio)) {
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
  return safeUploadAndDelete;
}

module.exports = { createMediaDelivery };
