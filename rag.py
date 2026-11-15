# Create vector embeddings of chunks of knowledge_base files and store them.
# For any query, find the relevant chunks based on the cosine similarity between the query embedding and vector embeddings.
# Expose a search function as facade to be used by the tools.
# Configure new tool call, it should use our search function and concatenate the chunks information to be passed to LLM
# Update System prompt as well to instrcut the LLM to use the tool to get the company policies, etc.

import os
import numpy
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")


def create_embedding(content):
    response = client.embeddings.create(input=content, model=os.getenv("EMBED_MODEL"))
    return response.data[0].embedding


def cosine_similarity(A, B):
    A_B = numpy.dot(A, B)
    A_len = numpy.linalg.norm(A)
    B_len = numpy.linalg.norm(B)

    return A_B / (A_len * B_len)


def save_vector_store(chunks):
    with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)
    print(f"Saved {len(chunks)} in vector store file")


def load_documents_and_embeddings():
    chunks = []
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            print(f"Loaded {len(chunks)} from vector store file")
            return chunks

    files = os.listdir(os.getenv("KB_PATH"))

    for file in files or []:
        if not file.endswith(".txt"):
            continue
        file_path = os.path.join(os.getenv("KB_PATH"), file)
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            file_chunks = file_content.split("\n\n")
            for index, file_chunk in enumerate(file_chunks or []):
                if not file_chunk:
                    continue
                file_chunk_embedding = create_embedding(file_chunk)
                chunks.append(
                    {
                        "source": file,
                        "content": file_chunk,
                        "embedding": file_chunk_embedding,
                        "size": len(file_chunk),
                        "chunkId": index,
                    }
                )
    save_vector_store(chunks)
    return chunks


def search_query_chunks(query, top_k=3):
    chunks = load_documents_and_embeddings()
    results = []
    query_embedding = create_embedding(query)
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        results.append(
            {
                "source": chunk["source"],
                "content": chunk["content"],
                "embedding": chunk["embedding"],
                "size": chunk["size"],
                "chunkId": chunk["chunkId"],
                "score": score,
            }
        )
    results.sort(key=lambda rec: rec["score"], reverse=True)
    return results[:top_k]
