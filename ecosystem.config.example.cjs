// Copy to ecosystem.config.cjs on the deployment host after backing up the old config.
// Credentials belong in that host's .env, which is loaded by the worker and bot.
const path = require('node:path');
const root = __dirname;
const common = {
  script: 'service_worker.py',
  interpreter: path.join(root, '.venv', 'bin', 'python'),
  cwd: root,
  instances: 1,
  exec_mode: 'fork',
  autorestart: true,
  watch: false,
  exp_backoff_restart_delay: 1000,
  kill_timeout: 15000,
  env: { NODE_ENV: 'production', PYTHONUNBUFFERED: '1' },
  log_date_format: 'YYYY-MM-DD HH:mm:ss',
};
module.exports = {
  apps: [
    { ...common, name: 'MediaGrabber', args: ['bot'],
      out_file: './logs/out.log', error_file: './logs/err.log' },
    { ...common, name: 'bns_daily',
      args: ['bns', '--start-at', '12:00', '--state-file', '.state/bns_state.json'],
      out_file: './logs/bns_out.log', error_file: './logs/bns_err.log' },
    { ...common, name: 'bns_daily_audio',
      args: ['bns', '--start-at', '08:00', '--target', '3744810661', '--audio-only',
        '--state-file', '.state/bns_audio_state.json'],
      out_file: './logs/bns_audio_out.log', error_file: './logs/bns_audio_err.log' },
    // Moments with Mahant Swami: posted at 08:00, available by 08:15, imported from 08:20.
    { ...common, name: 'moments_daily',
      args: ['moments', '--start-at', '08:20', '--state-file', '.state/moments_state.json'],
      out_file: './logs/moments_out.log', error_file: './logs/moments_err.log' },
  ],
};
