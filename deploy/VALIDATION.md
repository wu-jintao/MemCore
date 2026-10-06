# Public deployment verification status

Recorded October 5–6, 2026. This document describes development measurements,
not an official AML Smoke or Full. Deployment addresses, credentials and private
operational records are omitted. Source versions are distinguished in
[SOURCE_STATUS.md](../SOURCE_STATUS.md).

The earlier E5 release passed actual Ubuntu-host interface checks and a
10,000-message/16-caller synthetic semantic load, including 500 immediate top-1
visibility checks. Those findings do not validate newer source or Full-scale
storage, bandwidth and model capacity. Public health/authentication successes
also do not imply that full Add/Search responses reach clients reliably.

Isolated delayed-body and slow-reader controls reproduced Caddy's frontend idle
cutoffs. Explicit 30-minute body-read and write-idle settings were validated and
reloaded with backups. Upstream timeouts alone did not cover the frontend fields.
These changes repair two measured timeout behaviors; subsequent public transport
failures remain unresolved.

The final bounded low-concurrency sample planned 32 requests and exhausted its
600-second budget after 17 completions: **15 passes, 2 failures and 15 incomplete**.
The serial Search and identical-retry blocks each passed 7/8. No stable
concurrency, valid VPN-disconnected comparison or Full readiness is established.
Network causation is unresolved. Earlier failed and partial samples remain in
[the aggregate diagnostics](../competition/results/PUBLIC_TRANSPORT_DIAGNOSTICS.json).

Use `verify_public_api.py` only against an authorized deployment with its token
supplied privately. It checks schemas, retries, visibility and scope isolation
with generated data; it never substitutes for an official Smoke or capacity run.
For every probe retain failures/incomplete requests and source/configuration
identities. Final API behavior and model requirements must be verified again on
the exact version submitted for formal evaluation.
