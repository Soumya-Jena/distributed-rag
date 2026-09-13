# Apache Kafka Overview

Apache Kafka is a distributed event-streaming platform used to publish, store, and process ordered streams of records. Producers write events to topics, and consumers read events from those topics.

A topic is divided into partitions. Events within a partition have a stable order and receive sequential offsets. Partitions allow Kafka to distribute storage and processing across multiple brokers and enable consumer applications to process data in parallel.

Kafka brokers form a cluster. Each partition is replicated across brokers according to its replication factor. One replica acts as the leader and handles reads and writes, while follower replicas copy the partition data. If a broker fails, an eligible follower can become the new leader.

Consumers commonly belong to a consumer group. Kafka assigns each partition to at most one consumer within the group, allowing the group to divide the topic's workload. Different consumer groups can independently read the same events for separate applications.

Kafka retains events for a configured period or until a size limit is reached, regardless of whether consumers have already read them. Consumers track offsets and can replay older records when required.

Kafka replication concerns event-log partitions across Kafka brokers. It is conceptually related to fault tolerance and data distribution but is different from PostgreSQL streaming replication, which sends database WAL records from a primary server to standby servers.
