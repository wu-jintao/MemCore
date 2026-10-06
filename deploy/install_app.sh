#!/usr/bin/env bash
# Run on the participant's independent Ubuntu cloud instance as root.
# Format/mount the separately verified empty data disk before running this file.
set -euo pipefail
aml_source_dir=${1:?Usage: install_app.sh SOURCE_DIRECTORY VERIFIED_PUBLIC_HOSTNAME}
aml_api_host=${2:?A verified public hostname is required}
if [[ ! "$aml_api_host" =~ ^[a-zA-Z0-9.-]+$ ]]; then
  echo "Invalid public hostname" >&2
  exit 2
fi
if ! mountpoint -q /srv/aml-data; then
  echo "Refusing to store evaluation data on an unmounted system-disk directory" >&2
  exit 2
fi
if [ ! -f "$aml_source_dir/server.py" ] || [ ! -f "$aml_source_dir/semantic.py" ]; then
  echo "Missing reviewed server.py or semantic.py" >&2
  exit 2
fi
if ! id aml-memory >/dev/null 2>&1; then
  useradd --system --home-dir /srv/aml-data --shell /usr/sbin/nologin aml-memory
fi
install -d -m 0755 /opt/aml-memory/app /etc/aml-memory
install -d -o aml-memory -g aml-memory -m 0700 /srv/aml-data/db /srv/aml-data/cache
install -d -m 0755 /srv/aml-data/models
install -m 0644 "$aml_source_dir/server.py" /opt/aml-memory/app/server.py
install -m 0644 "$aml_source_dir/semantic.py" /opt/aml-memory/app/semantic.py
if [ ! -x /opt/aml-memory/venv/bin/python ]; then
  python3 -m venv /opt/aml-memory/venv
fi
if [ ! -f /etc/aml-memory/runtime.env ]; then
  python3 - <<'PY'
import os
import secrets
path = '/etc/aml-memory/runtime.env'
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, 'w') as stream:
    stream.write('MEMORY_API_TOKEN=' + secrets.token_urlsafe(48) + '\n')
    stream.write('HF_HOME=/srv/aml-data/cache/huggingface\n')
PY
fi
install -m 0644 "$aml_source_dir/deploy/aml-memory.service" /etc/systemd/system/aml-memory.service
sed "s/^API_HOST /$aml_api_host /" "$aml_source_dir/deploy/Caddyfile.template" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl enable aml-memory
systemctl restart aml-memory
systemctl reload caddy
systemctl is-active aml-memory caddy
