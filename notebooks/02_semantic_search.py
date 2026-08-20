from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "PostgreSQL supports streaming replication using WAL records.",
    "PostgreSQL uses MVCC to provide transaction isolation.",
    "PostgreSQL indexes can improve query performance.",
    "Kubernetes schedules containers onto available nodes.",
    "Kubernetes provides service discovery and load balancing.",
    "Redis is an in-memory key-value data store.",
    "Kafka is a distributed event streaming platform.",
]

query = "How does PostgreSQL replicate data to another server?"

document_embeddings = model.encode(documents)
query_embedding = model.encode([query])

scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

results = sorted(
    zip(scores, documents),
    reverse=True
)

print("\nQuery:")
print(query)

print("\nResults:")

for score, document in results:
    print(f"{score:.4f} | {document}")