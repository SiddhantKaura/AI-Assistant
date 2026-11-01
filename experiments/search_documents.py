import os
import numpy
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    # Not needed as we are hitting local LLM, adding dummy key since OpenAI has made it non null
    api_key=os.getenv("API_KEY"),
)


def get_embedding(text):
    response = client.embeddings.create(input=text, model=os.getenv("EMBED_MODEL"))
    return response.data[0].embedding


def load_documents(folder):
    documents = []
    files = os.listdir(folder)
    for file_name in files:
        if not file_name.endswith(".txt"):
            continue
        file = os.path.join(folder, file_name)
        with open(file, "r", encoding="utf-8") as file_content:
            documents.append({"name": file_name, "content": file_content.read()})
    # print(documents)
    return documents


def load_embeddings(documents):
    for document in documents:
        document["embedding"] = get_embedding(document["content"])
        # print(document)


def cosine_similarity(embedding1, embedding2):
    dot_prod = numpy.dot(embedding1, embedding2)
    embedding1_len = numpy.linalg.norm(embedding1)
    embedding2_len = numpy.linalg.norm(embedding2)

    return dot_prod / (embedding1_len * embedding2_len)


def search_query(query, documents, top_k=3):
    query_embedding = get_embedding(query)
    results = []
    for document in documents:
        score = cosine_similarity(query_embedding, document["embedding"])
        results.append(
            {
                "score": score,
                "source": document["name"],
                "text": document["content"],
                "size": len(document["content"]),
            }
        )
    results.sort(key=lambda rec: rec["score"], reverse=True)
    return results[:top_k]


documents = load_documents(os.getenv("KB_PATH"))
load_embeddings(documents)
results = search_query(
    query="What is the maternity leave policy ?", documents=documents
)
for result in results:
    print(f"{result['score']} - {result['source']} - {result['size']}")
