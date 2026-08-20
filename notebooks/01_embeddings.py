from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "PostgreSQL supports streaming replication.",
    "PostgreSQL can replicate WAL records to standby servers.",
    "Kubernetes manages containerized applications.",
    "The weather is sunny today."
]

embeddings = model.encode(sentences)

similarity = cosine_similarity(embeddings)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(
            f"{i} ↔ {j}: "
            f"{similarity[i][j]:.4f}"
        )