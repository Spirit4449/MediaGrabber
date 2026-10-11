# Production update

After copying/pulling the updated code onto the Linux server:

1. Keep your existing `.env`. Add/update only these settings:

   ```dotenv
   BNS_RUN_TIMEOUT_SECONDS=1800
   BNS_ALERT_AFTER=18:00
   ```

2. Replace the PM2 definitions and restart:

   ```bash
   cd /home/nisch/Desktop/MediaGrabber
   cp ecosystem.config.cjs ecosystem.config.before-recovery.cjs
   pm2 stop MediaGrabber bns_daily bns_daily_audio
   cp ecosystem.config.example.cjs ecosystem.config.cjs
   pm2 delete MediaGrabber bns_daily bns_daily_audio
   pm2 start ecosystem.config.cjs
   pm2 save
   pm2 status
   ```

Your existing production `.env`, sessions, checkpoints, and personal-chat notification setting stay in place. No installation needed. If reboot startup was never enabled, run `pm2 startup` and its printed command once.

**Result:** Audio starts checking at 8 a.m.; normal BNS starts at noon. Each checks every 15 minutes until today's BNS is successfully delivered or 6 p.m., then resumes the next day. Times use `BNS_TIMEZONE` (default `America/New_York`). Each attempt is limited to 30 minutes. At 6 p.m., missing delivery triggers an alert to your existing `TELEGRAM_NOTIFY_CHAT_ID` (once per BNS job per day).

An attempt already running at 6 p.m. can finish within its 30-minute limit; no new attempts start after 6. The alert still goes out if delivery is missing at the deadline. The bot watchdog remains active all day.
