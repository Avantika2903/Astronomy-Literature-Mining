from sklearn.feature_extraction.text import TfidfVectorizer
texts = []
for p in papers:
    combined = p["title"] + " " + p["abstract"]
    texts.append(combined)
vectorizer = TfidfVectorizer(
    stop_words='english',
    token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
)
tfidf_matrix = vectorizer.fit_transform(texts)
tfidf_matrix
print(vectorizer.get_feature_names_out()[:50])

#COSINE SIMILARITY
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
query = "machine learning astronomy applications"
query_vec = vectorizer.transform([query])
scores = cosine_similarity(query_vec, tfidf_matrix)
top_indices = np.argsort(scores[0])[::-1][:5]
for i in top_indices:
    print("TITLE:", papers[i]["title"])
    print("SCORE:", scores[0][i])
    print("-"*50)
