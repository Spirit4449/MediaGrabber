const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { spawn } = require("node:child_process");
const { consumeDownloader } = require("./downloader-events");
const { fmtMB, progressText } = require("./progress");

const INVITE_WAIT_MS = 5 * 60 * 1000;

function createDownloadFlow({ rootDir, pythonBin, safeUploadAndDelete, spawnWorker, consumeWorker = consumeDownloader }) {
  const state = new Map();
  function spawnDownloader(args) {
    console.log("[spawn] python downloader.py", args.join(" "));
    return spawn(
      pythonBin,
      [path.join(rootDir, "downloader.py"), ...args],
      { cwd: rootDir },
    );
  }
  async function handleDownloadFlow(ctx, link, opts = {}) {
    const DOWNLOAD_ROOT = path.join(rootDir, "downloads");
    await fs.promises.mkdir(DOWNLOAD_ROOT, { recursive: true });
    await ctx.reply(`🟢 Received link:\n${link}`);
    const message = await ctx.reply("📥 Connecting and checking media…");
    startBackgroundDownload(ctx, link, message.message_id, { invite: opts.invite, allowInvite: true });
  }

  function startBackgroundDownload(ctx, link, progressMsgId, opts = {}) {
    const chatId = ctx.chat.id;
    const args = ["--link", link, "--outdir", path.join(rootDir, "downloads", crypto.randomUUID())];
    if (opts.invite) args.push("--invite", opts.invite);
    const py = (spawnWorker || spawnDownloader)(args);
    let lastPctLogged = -10;
    let lastProgressEdit = 0;
    consumeWorker(py, {
      onFailure: async error => {
        console.error("[worker failure]", error.message);
        await ctx.telegram.editMessageText(chatId, progressMsgId, undefined,
          `❌ ${error.message}`).catch(() => {});
      },
      onEvent: async ev => {
        if (ev.type === "progress") {
          const pct = typeof ev.pct === "number" ? ev.pct : 0;
          const text = pct >= 100 ? "✅ Download complete. Finalizing file…" : progressText(ev.downloaded, ev.total, pct);
          const now = Date.now();
          if (now - lastProgressEdit >= 2000 || pct >= 100) {
            lastProgressEdit = now;
            await ctx.telegram.editMessageText(chatId, progressMsgId, undefined, text).catch(() => {});
          }
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
          if (opts.allowInvite && !opts.invite) {
            state.set(chatId, { awaitingInviteFor: link, expiresAt: Date.now() + INVITE_WAIT_MS });
            await ctx.telegram.editMessageText(chatId, progressMsgId, undefined,
              "🔐 I need an invite link to join that channel/group.\nPlease send: https://t.me/+INVITEHASH\nType stop to cancel. (Expires in 5 minutes.)").catch(() => {});
          } else {
            await ctx.telegram.editMessageText(chatId, progressMsgId, undefined,
              "❌ No access to that post. Please send the post link again with a valid invite.").catch(() => {});
          }
        }
      },
    });
    py.stderr.on("data", chunk => console.error("[downloader stderr]", chunk.toString("utf8")));
  }

  return { handleDownloadFlow, startBackgroundDownload, state };
}

module.exports = { createDownloadFlow };
