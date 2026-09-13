# PostgreSQL Streaming Replication

PostgreSQL streaming replication keeps one or more standby servers synchronized with a primary database server. The primary records every database change in the write-ahead log, commonly called WAL. A standby connects to the primary and continuously receives these WAL records through PostgreSQL's replication protocol.

The standby replays the WAL records to reproduce the changes made on the primary. By default, streaming replication is asynchronous: the primary can commit a transaction before a standby confirms that it received the corresponding WAL. This provides low transaction latency, but a failure of the primary can result in the loss of recently committed transactions if those records had not reached the standby.

Synchronous replication can require one or more standby servers to acknowledge a transaction before the primary reports that transaction as committed. This improves durability but increases commit latency because the primary must wait for a remote acknowledgement.

A standby can be promoted when the primary becomes unavailable. Promotion ends recovery mode and allows the standby to accept writes as the new primary. PostgreSQL does not provide a complete automatic failover manager by itself, so production systems often use an external orchestration tool to detect failures, promote a standby, and redirect clients.

Replication slots can prevent the primary from deleting WAL segments that a standby or logical replication consumer still needs. Slots must be monitored carefully because an inactive consumer can cause retained WAL to consume significant disk space.

Physical streaming replication copies changes for the entire PostgreSQL cluster at the storage level. Logical replication instead publishes row-level changes for selected tables and can replicate between databases with different physical layouts or major PostgreSQL versions.
