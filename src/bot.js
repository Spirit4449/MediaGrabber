const { isAllowedTelegramLink, isInviteLink } = require("./telegram-links");

function registerBot(bot, { state, handleDownloadFlow }) {
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

}

module.exports = { registerBot };
