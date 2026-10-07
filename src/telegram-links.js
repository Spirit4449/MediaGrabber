function isAllowedTelegramLink(link) {
  return /^https?:\/\/t\.me\/(?:c\/\d+\/\d+|[A-Za-z0-9_]+\/\d+)$/.test(link);
}
function isInviteLink(text) {
  return (
    /t\.me\/(?:\+|joinchat\/)[A-Za-z0-9_-]+$/.test(text.trim()) ||
    /^[A-Za-z0-9_-]{16,}$/.test(text.trim())
  );
}
module.exports = { isAllowedTelegramLink, isInviteLink };
