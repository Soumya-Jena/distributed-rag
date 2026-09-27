# Corpus scalability runbook

The commands below must be run from the repository root with Docker PostgreSQL running.

```powershell
python -m evaluation.setup_scale_db
python -m evaluation.generate_scale_data --targets 10000
python -m evaluation.evaluate_corpus_scale --stage S1 --expected-chunks 10000
python -m evaluation.generate_scale_data --targets 50000
python -m evaluation.evaluate_corpus_scale --stage S2 --expected-chunks 50000
```

Repeat for 100000, 250000, 500000, and 1000000 only while the safety gate remains healthy. Generate the chart after the last healthy tier:

```powershell
python -m evaluation.plot_corpus_scale
```

## Isolation and reproducibility

- `DATABASE_URL_SCALE` must end in `/ragdb_scale`; the code rejects any other database.
- `ragdb` is never mutated by the scalability scripts.
- Synthetic vectors use seed 42 and are streamed in batches of 500.
- `scale_ingestion_runs` records progress, status, and the last inserted row count.
- Re-running a completed target does not duplicate rows.
- The dataset manifest is at `datasets/scaling/manifest.json`.

## Safe-stop policy

Stop before the next tier when host memory is at least 90%, free disk is below 10 GiB, PostgreSQL is unstable, a query exceeds its 30-second statement timeout, or the machine begins sustained swapping. Preserve the largest healthy tier as the measured ceiling; a safe stop is a result, not a reason to change the thresholds after seeing results.

