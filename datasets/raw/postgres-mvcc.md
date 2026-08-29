# PostgreSQL MVCC

PostgreSQL uses multiversion concurrency control, or MVCC, to allow transactions to read and write data concurrently. Instead of overwriting a row in place, an update creates a new row version. Older row versions can remain visible to transactions whose snapshots were created before the update.

Each SQL statement or transaction observes a snapshot that determines which row versions are visible. Under the Read Committed isolation level, each statement receives a new snapshot. Under Repeatable Read, the transaction continues to use the same snapshot, so repeated queries see a stable view of committed data.

MVCC reduces contention because readers generally do not block writers and writers generally do not block readers. Row-level locks are still used when concurrent transactions attempt incompatible modifications to the same rows.

Deleted and obsolete row versions occupy storage until PostgreSQL determines that no active transaction can see them. The VACUUM process marks this space reusable. Autovacuum runs automatically based on table activity and is essential for controlling table bloat and preventing transaction ID wraparound.

MVCC provides visibility rules and isolation; it is not a replication mechanism. Streaming replication uses WAL records to transfer database changes to standby servers, while MVCC controls how concurrent transactions on a PostgreSQL server observe row versions.
