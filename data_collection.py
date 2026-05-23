pip install arxiv
import arxiv
client = arxiv.Client()

search = arxiv.Search(
    query="(machine learning OR deep learning) AND (astronomy OR astrophysics OR cosmology)  AND submittedDate:[20240101 TO 20260430]",
    max_results=100
)
papers = []
for result in client.results(search):
    papers.append({
        "title": result.title,
        "abstract": result.summary
    })
print(len(papers))
for i in range(5):
    print("TITLE:", papers[i]["title"])
    print("ABSTRACT:", papers[i]["abstract"])
    print("-"*50)
