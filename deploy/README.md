# Isolated operations tools

These are optional operator-run utilities. No service address, token, model key,
SSH configuration or deployed state is included. `prepare_http_candidate.py`
prepares an isolated Linux/systemd v4 service from an explicitly reviewed
exact-two-file manifest and separately staged private model configuration.
It requires root on the intended host. Its `prepare`, `start`, `stop` and `status`
commands do not publish a route or modify the existing production application.
Review its fixed filesystem layout and service account assumptions before use.
The supplied `CORE_MANIFEST.json` has the exact format this tool accepts; pass
its independently checked SHA-256 with `--manifest-sha256`.

`probe_http_candidate.py` is dry-plan by default. With `--execute`, it reads only
`MEMORY_API_TOKEN` from the supplied private literal file and sends a fixed small
synthetic loopback probe: one 32-message Add, identical retry, conflicting retry,
16 concurrent top100 Searches and exact-user-isolation checks. The maximum is
30 HTTP requests, including health, with explicit timeout and wall deadline.
It refuses redirects and environment proxies, preserves planned/completed/
succeeded/failed/unfinished counts and does not publish response bodies or user
IDs. This is a functional sample, not a Full-scale capacity qualification.

Example plan (does not read the token file or send requests):

```sh
python3 deploy/probe_http_candidate.py --base-url http://127.0.0.1:18081   --token-env-file /path/to/private-service.env   --output .local/synthetic-probe.json --deadline-seconds 180
```

Add `--execute` only on the intended authorized loopback host with a private
file. `verify_public_api.py` is a separate HTTPS transport diagnostic. Its CLI
executes requests directly after validating arguments and reading the service
token from its environment; it has no dry-run/`--execute` flag or source-manifest
guard. It is not an official AML evaluator. Review `--help` and fixed
request/denominator limits before using it. The historical production activation
script and host-specific Caddy/systemd configuration are intentionally excluded.

Tests use local mocks and temporary private files. Do not include runtime
reports, host inventory or credentials in a source publication.

The v1.1 queuefix preparation validator accepts HTTP queue timeouts up to 900
seconds. The selected deployment profile is explicitly queue 900 / operation
600; no operation-clock behavior was changed. Local/disabled embedding queue
bounds remain 300 seconds. The generic operation upper bound remains unchanged,
so the combined 1,500-second budget requires retaining the documented 600-second
operation setting rather than choosing an arbitrary larger operation timeout.
