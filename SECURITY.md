# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/Praneetha1844/Driftwatch/security/advisories/new)
for this repository. Do not open a public issue for a security problem. Include what you
found, how to reproduce it and which version or commit it affects.

This is a personal project maintained by one person. There is no service level agreement,
but reports are read and answered as soon as practical.

## Scope

In scope: the code in this repository, its Docker and Compose files and the default settings.

Out of scope: vulnerabilities in third-party dependencies (report them upstream), and
problems in a deployment's own configuration.

## Status and known limits

Driftwatch has not been run in production or independently reviewed. Points that matter for
security:

- **Pickle-based artifacts.** Models are stored with `joblib`, which uses pickle. Loading a
  malicious artifact can execute code. Only load artifacts you produced, and protect the
  registry directory from untrusted writers.
- **No authentication** on the serving API. Keep it on an internal network.
- **Request data is buffered in memory** for drift checks and is lost on restart. Do not send
  data you are not allowed to hold in process memory.
- **Registry names** are restricted to letters, digits, `_`, `.` and `-`, starting with a
  letter or digit, so a name cannot leave the registry root.
- Dependencies are lower-bounded, not pinned, and not scanned automatically.

## Handling secrets

Never commit `.env` files or credentials. The Compose file reads `GRAFANA_ADMIN_PASSWORD` from
the environment. If a secret is committed by mistake, rotate it first, then remove it.
