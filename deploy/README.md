# Independent Ubuntu deployment

These templates implement a private application and database behind public
HTTPS. They are not proof that a host is deployed or sufficient for AML Full.

`activate_semantic.py` pins the source hashes of the candidate verified and
activated on October 5, 2026. Later local source changes are new candidates,
and this script will reject them until a separately reviewed source manifest
and capacity validation are ready. Preserve the existing activation report and
recovery artifacts when preparing the next release.

1. Authenticate over SSH using the participant's existing downloaded private key.
   Verify the host, CPU, RAM, OS, block devices, mount points and listening ports
   using `inspect_host.py`. Never log or copy the private key into this repository.
2. Inspect the separate data device and its partition table, filesystem signatures
   and mounts. Format only a positively identified new empty non-system disk.
   If any data/signature exists, preserve it and resolve its intended use first.
   Mount it at `/srv/aml-data`, persist its UUID in `/etc/fstab`, and verify the
   actual mount. `install_app.sh` refuses an unmounted data directory.
3. Install Python venv support and Caddy from the official signed Ubuntu/Debian
   package source described in the [Caddy install guide](https://caddyserver.com/docs/install).
   Inspect the current version and validate the Caddy template on that version.
4. Copy reviewed source and the deploy directory to a private staging directory.
   Verify a public hostname resolves to this instance's public IP. A hostname
   from an embedded-IP DNS service is usable for an initial test but adds an
   external availability dependency; it is not an owned domain.
5. As root, run `install_app.sh SOURCE_DIRECTORY VERIFIED_PUBLIC_HOSTNAME`.
   It creates an unprivileged service user, private database/cache directories,
   a venv and a random service token only when no environment file exists.
   The token lives in `/etc/aml-memory/runtime.env` with mode 0600 and is never
   printed. systemd reads the file before dropping privileges.
6. Verify Caddy's trusted TLS certificate and outside-network `/health`, 401 for
   unauthenticated POST, authenticated Add/Search, identical retries and exact
   complete-user isolation. Validate database persistence after service restart.
   Run measured cloud and public-network load at the intended evaluation
   concurrency with generated data, then start the accepted version from a
   fresh test-free database or remove test scopes using an audited process.

`aml-memory.service` binds to 127.0.0.1:8080, requires the data mount, restarts on
failure, limits writable paths and uses a 12GiB memory cap. `Caddyfile.template`
allows 8MiB requests, 30-minute frontend body-read/write idle timeouts and a
30-minute upstream response timeout. Caddy v2.11.7 otherwise defaults both idle
deadlines to 60 seconds; upstream transport timeouts do not override them. The
`servers :443` / `timeouts` options `read_body_idle 30m` and `write_idle 30m`
were validated on that version and in separate delayed-body/slow-reader tests.
The header deadline retains its default; the isolated plain-HTTP header test
does not establish the cause of observed TLS failures. Inspect the adapted JSON
on another Caddy version before installation. Only the API
ports require public access; the application port and database are private.
HTTP is redirected to HTTPS by Caddy. Do not expose a company development host
or use an internal model/database gateway in the formal request path.

The current Caddy template also writes bounded private HTTP status/timing logs
to `/var/log/caddy/aml-memory/access.jsonl` (0700 directory, 0600 file). It removes
the complete request object, response headers and user identifier; it records
method and path without query, upstream timing, response status/bytes and total
duration. It does not collect request bodies. Ten-MiB rotation retains up to
three backups or 72 hours. HTTP logs cannot observe failures before HTTP handling
or prove that a client received the complete response.

After trusted TLS and authentication are installed, run the outside-network
synthetic probe from the participant's computer:

```sh
python3 deploy/verify_public_api.py --base-url https://VERIFIED_HOSTNAME --concurrency 16
```

Supply `MEMORY_API_TOKEN` through the private environment. The probe never prints
it, follows no redirects and uses default certificate validation. It writes a
local measured report, checks public health/auth, concurrent Add and immediate
visibility, repeated/conflicting requests and exact complete-user isolation.
Its 320 generated messages are a small network probe, not formal-scale capacity
or an official Smoke. Long-history load and recovery are separate gates.

The initial install runs the default lexical backend. Enabling the local model
requires a separately downloaded, revision-verified directory on the persistent
disk and an isolated CPU dependency installation in the service venv. Set the
reviewed embedding environment before creating its fresh database. Verify model
fingerprint, latency, queueing, disk growth and recovery on the actual host.
Changing model/configuration after an accepted Full is prohibited.

The measured Ubuntu 22.04 / Python 3.10 x86_64 CPU environment is recorded in
`requirements-semantic-linux-python310.lock.txt`, including the official hashed
CPU PyTorch wheel. Install this lock only in the isolated Linux service venv;
the local Mac environment has a different platform and is not a substitute.
The prepared cloud model and dependency checks are recorded in `VALIDATION.md`.
Neither the lock nor a constrained test proves capacity on an actual resized
instance; verify the final host and service configuration before submitting.

The API token is distinct from the AML evaluation access key. Configure it as a
Bearer credential at the authorized AML destination after reviewing the access
request; do not expose it in URLs, documentation, screenshots, Git or logs.
Do not rely on a machine name containing “48h” for an automatic stop. Pay-as-you-go
instances, retained disks and IP/bandwidth can continue accruing charges.
