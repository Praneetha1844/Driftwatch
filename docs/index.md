# Driftwatch documentation

| Document | Read it to |
|---|---|
| [README](../README.md) | Understand what Driftwatch is and run the demo |
| [Architecture](ARCHITECTURE.md) | See the lifecycle, components, artifact format and drift computation |
| [Command line](CLI.md) | Use `simulate`, `train`, `register`, `promote`, `rollback`, `drift`, `decide` and `serve` |
| [Configuration](CONFIGURATION.md) | Look up every setting and environment variable |
| [Serving API](API.md) | Call `/predict`, `/drift`, `/health` and `/metrics` |
| [Runbook](RUNBOOK.md) | Act on drift reports, refused promotions and rollbacks |
| [Development guide](DEVELOPMENT.md) | Set up, test and extend the code |
| [Model card template](../MODEL_CARD.md) | Document a model you actually train |
| [Security policy](../SECURITY.md) | Report a vulnerability and read the known limits |
| [Decision records](adr) | Understand why the design is the way it is |

## Decision records

1. [PSI and KS together](adr/0001-psi-and-ks-together.md)
2. [Gated promotion](adr/0002-gated-promotion.md)
3. [File-based fallbacks](adr/0003-file-based-fallbacks.md)
4. [Seeded synthetic data](adr/0004-seeded-synthetic-data.md)
