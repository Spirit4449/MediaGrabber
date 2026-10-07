// Copy to ecosystem.config.cjs and choose the host's schedule and channels.
const path = require('node:path');
const root = __dirname;

module.exports = {
  apps: [
    {
      name: 'satsangfetcher',
      script: 'server.js',
      cwd: root,
      instances: 1,
      autorestart: true,
      watch: false,
      env: { NODE_ENV: 'production' },
    },
    {
      name: 'bns_daily',
      script: 'daily_bns_sync.py',
      interpreter: path.join(root, '.venv', 'bin', 'python'),
      cwd: root,
      instances: 1,
      autorestart: false,
      watch: false,
      // Example: 17:00 in the deployment host's timezone.
      // PM2 also starts this script immediately when the app is first started.
      cron_restart: '0 17 * * *',
    },
  ],
};
