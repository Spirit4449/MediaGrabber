const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const rateLimit = require("express-rate-limit");
const { isAllowedTelegramLink } = require("./telegram-links");

function registerApi(app, { bot, sharedSecret, rootDir, startBackgroundDownload }) {
  const limiter = rateLimit({ windowMs: 60_000, max: 30 });
  app.use("/api/download", limiter);

  function verifyHmac(req, res, next) {
    const sig = req.get("x-signature") || "";
    const body = JSON.stringify(req.body || {});
    const mac = crypto
      .createHmac("sha256", sharedSecret)
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

      const DOWNLOAD_ROOT = path.join(rootDir, "downloads");
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

  app.get("/healthz", (_req, res) => res.json({ ok: true }));
}

module.exports = { registerApi };
