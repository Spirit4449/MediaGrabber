# syntax=docker/dockerfile:1
# MediaGrabber: the Telegram bot (Node server.js + Python downloader) and the daily jobs
# (service_worker.py bns|moments). Started by the Brahma naa Sange compose.yaml. No browser:
# BNS uploads are plain HTTP (bns_site.py).
FROM node:24.21.0-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg tzdata ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements.txt
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY . .

USER node
ENV NODE_ENV=production PYTHONUNBUFFERED=1
CMD ["/app/.venv/bin/python", "service_worker.py", "bot"]
