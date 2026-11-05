# Security

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub's security advisories](https://github.com/HawkBearPig/dgpp/security/advisories/new)
rather than in a public issue. Include the server version from
`dgpp-serve --version`, the deployment shape and a reproduction. Reports
are acknowledged within a week, and the fix and its disclosure are
coordinated with the reporter.

## Supported versions

Fixes land on `master`. Releases are versioned tarballs built from it, and
the latest release is the one that receives fixes.

## What the server assumes

`dgpp-serve` has no authentication and no TLS. It is meant for a trusted
network. Bind it to `127.0.0.1` and put a reverse proxy in front of it for
anything else. The [nginx example](deploy/nginx.example.conf) terminates
TLS and adds basic authentication. Ranks coordinate over TCP and RoCE on
the fabric without authentication, so the fabric must be private to the
deployment.

Checkpoints are executed data. The chat template is interpreted by the
engine's own Jinja implementation, and request lengths, tool schemas and
JSON Schema constraints are validated and bounded before dispatch. Serve
checkpoints from sources you trust, and keep the Hugging Face cache
writable only by the serving user.
