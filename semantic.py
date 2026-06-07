
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import arxiv

# -----------------------------
# STEP 1: Collect arXiv papers
# -----------------------------

client = arxiv.Client()

search = arxiv.Search(
    query="(machine learning OR deep learning) AND (astronomy OR astrophysics OR cosmology) AND submittedDate:[20240101 TO 20260430]",
    max_results=100
)

papers = []

for result in client.results(search):
    papers.append({
        "title": result.title,
        "abstract": result.summary
    })

print(f"Total papers collected: {len(papers)}")

# ----------------------------------------
# STEP 2: Combine title + abstract text
# ----------------------------------------

paper_corpus = []

for paper in papers:
    combined_text = paper["title"] + " " + paper["abstract"]
    paper_corpus.append(combined_text)

# ----------------------------------------
# STEP 3: Load Sentence Transformer model
# ----------------------------------------

model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------------------
# STEP 4: Generate paper embeddings
# ----------------------------------------

paper_embeddings = model.encode(
    paper_corpus,
    convert_to_numpy=True
)

print("Embedding shape:", paper_embeddings.shape)

# ----------------------------------------
# STEP 5: User query
# ----------------------------------------

query = "machine learning methods for galaxy analysis"

# Convert query into embedding
query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

# ----------------------------------------
# STEP 6: Compute cosine similarity
# ----------------------------------------

similarity_scores = cosine_similarity(
    query_embedding,
    paper_embeddings
)

# ----------------------------------------
# STEP 7: Retrieve top matching papers
# ----------------------------------------

top_indices = np.argsort(similarity_scores[0])[::-1][:5]

print("\nTop Semantic Search Results:\n")

for idx in top_indices:
    print("TITLE:", papers[idx]["title"])
    print("SIMILARITY SCORE:", round(similarity_scores[0][idx], 4))
    print("-" * 80)
