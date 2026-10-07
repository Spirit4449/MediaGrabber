# Move services to brobattles.dev

These are proposed production configuration files based on the supplied nginx.conf.
They have not been applied on the production device or validated by its Nginx binary.

## Routes

| New hostname | Existing backend |
| --- | --- |
| brobattles.dev, www.brobattles.dev | 192.168.1.97:3002 (game) |
| game.brobattles.dev | 192.168.1.97:3002 |
| classchats.brobattles.dev | 192.168.1.97:3000 |
| mc.brobattles.dev | 192.168.1.97:8800 |
| grades.brobattles.dev | 192.168.1.97:3001 |
| bns.brobattles.dev | 192.168.1.97:3003 |
| ops.brobattles.dev | 192.168.1.97:5173 |

Existing classchats.net routes remain during transition. `/api/download` continues
to use 127.0.0.1:4000 on the Nginx device, as in the supplied configuration.
MediaGrabber and Nginx share the production device. Both BNS domains serve originals
from `/var/www/mediagrabber-downloads/`. Create this directory with the bot user as
owner (mode 0755) and set `SHARED_DOWNLOAD_DIR` to that path; see public-downloads.md.
Other hostnames return 404 for `/downloads/`.

## DNS

The supplied Cloudflare DNS export already contains bns, mc, grades, ops and www.
Add CNAME records with names `classchats` and `game`, target `brobattles.dev`, matching
the existing service records' proxy setting. Leave unrelated mail, play, anirdesh,
and test records alone. DNS changes must be made in the authoritative DNS service;
this export is not a live configuration.

## Apply on production

1. Back up the active Nginx configuration, retaining a copy for rollback.
2. Apply `nginx-domain-migration-bootstrap.conf` to the active main configuration
   (the supplied file appears to be /etc/nginx/nginx.conf; confirm your active file).
   This adds the new names to the HTTP listener and moves HTTP redirects into
   `location /` so the ACME challenge location can be served. Existing HTTPS routing
   remains as supplied.
3. Run `sudo nginx -t`, then `sudo systemctl reload nginx` only if validation succeeds.
4. Confirm `/var/www/letsencrypt` exists and HTTP port 80 challenge requests reach
   the Nginx webroot. Cloudflare redirects or other rules must not block challenges.
   Expand the existing certificate:

```sh
sudo certbot certonly --webroot -w /var/www/letsencrypt \
  --cert-name brobattles.dev --expand \
  -d brobattles.dev \
  -d www.brobattles.dev \
  -d classchats.brobattles.dev \
  -d mc.brobattles.dev \
  -d grades.brobattles.dev \
  -d game.brobattles.dev \
  -d bns.brobattles.dev \
  -d ops.brobattles.dev
```

If the existing brobattles.dev certificate covers additional names, include those
names too rather than dropping them. Inspect with `sudo certbot certificates` first.
If issuance fails, keep the bootstrap config and resolve validation before proceeding.
Do not remove certificate directives from a live SSL listener to work around issuance.

5. Apply `nginx-domain-migration.conf`, run `sudo nginx -t`, and reload only on success.
6. Open each new HTTPS hostname and exercise login and API requests. Apps may need
   their allowed origins, OAuth callbacks, cookie domains, redirects, and hardcoded
   URLs updated separately; Nginx routing cannot make those changes automatically.
7. Set the bot's download directory and new public URL, then restart PM2 with
   `pm2 restart MediaGrabber --update-env`:

```ini
SHARED_DOWNLOAD_DIR=/var/www/mediagrabber-downloads
PUBLIC_DOWNLOAD_BASE_URL=https://bns.brobattles.dev/downloads/
DOWNLOAD_RETENTION_HOURS=72
BOT_SEND_COMPRESSED_COPY=false
```

When retiring classchats.net, remove its HTTP/HTTPS server blocks and replace the
HTTPS default server's certificate paths: the supplied catch-all currently uses the
classchats.net certificate. Check Certbot renewal configuration as well. Old HTTPS
links cannot redirect reliably after their DNS/domain or certificate expires.

## Verification available here

Both generated files preserve all upstream addresses/ports from the supplied config.
Only the final configuration adds new HTTPS service routing. Nginx and production
certificate files are not available locally, so the mandatory `nginx -t` check must
be run on production. No DNS records, certificates, or production files were changed.
