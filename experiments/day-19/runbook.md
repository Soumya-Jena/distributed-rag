# Container deployment runbook

## Prerequisites

Create the ignored secret file before using Compose:

```powershell
New-Item -ItemType Directory -Force secrets
Set-Content secrets/postgres_password.txt "replace-with-a-strong-password"
```

The secret is mounted through `/run/secrets`; it is not stored in `.env`, the image, or Git.

## Start and migrate

```powershell
docker compose build
docker compose up -d postgres redis model-server
docker compose --profile tools run --rm db-migrate
docker compose up -d rag-api
python -m evaluation.verify_deployment
```

Use `docker compose -f docker-compose.yml -f compose.dev.yaml up -d` for local access to PostgreSQL on 5432, Redis on 6379, and the model server on 8010. Production intentionally exposes only API port 8000.

## Ingest

```powershell
docker compose --profile tools run --rm ingest
```

The development override mounts `datasets/` read-only for this job. For a deployment, supply an equivalent read-only dataset volume or build an ingestion-specific artifact.

## Health and diagnosis

`GET /live` only proves that the API process is serving requests. `GET /ready` checks PostgreSQL, embeddings, the generation service, and security validation. Redis is deliberately absent from readiness because cache loss is fail-open. The model server exposes the same `/live` and `/ready` distinction on the private network.

```powershell
docker compose ps
docker compose logs --tail 100 rag-api model-server
Invoke-RestMethod http://localhost:8000/live
Invoke-RestMethod http://localhost:8000/ready
```

If the API is live but unready, inspect the readiness check map and the model/PostgreSQL logs. Restart the failed hard dependency and wait for `/ready` to return 200. Do not route traffic while it returns 503.

## Persistence and rollback

`postgres_data`, `redis_data`, and `model_cache` are named volumes. Recreating a container does not delete them. Never use `docker compose down --volumes` unless permanent data deletion is intended and separately approved.

The pre-change application state is tagged `pre-container-architecture-v1`. Roll back application code/image to that tag while retaining the named database volume. Database migration is idempotent and additive in this iteration.

## Optional profiles

```powershell
docker compose --profile observability up -d
docker compose --profile resilience up -d toxiproxy
```

Stop the stack gracefully with `docker compose stop`; the application services have a 30-second grace period and an init process for signal forwarding.
