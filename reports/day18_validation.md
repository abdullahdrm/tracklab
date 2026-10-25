# TrackLab quality-gate validation

The repository quality gate covers pytest, Ruff lint, Ruff format checking,
Compose configuration validation and `git diff --check`. GitHub Actions runs
the same lint, format and test checks for pushes and pull requests targeting
`master`, with a Docker smoke job for the container health endpoint.

The FastAPI service remains intentionally scoped as a local demonstration. It
does not provide authentication, durable worker queues, production quotas or
production observability.
