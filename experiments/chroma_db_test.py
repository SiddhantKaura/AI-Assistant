import chromadb
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBED_MODEL")

openaiClient = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
# Persistent DB client to persist data on disk
chromadbClient = chromadb.PersistentClient(path="./chromadbStorage")


def create_embedding(content):
    response = openaiClient.embeddings.create(input=content, model=EMBEDDING_MODEL)
    return response.data[0].embedding


# Using cosine similarity for semantic matching
collection = chromadbClient.get_or_create_collection(
    name="NOVATECH_SOLUTIONS", configuration={"hnsw": {"space": "cosine"}}
)

chunks = {
    "c0": "Employees are expected to work 40 hours per week, typically from Monday to Friday. Standard working hours are 9:00 AM to 6:00 PM, with a one-hour lunch break. Employees must inform their managers if they expect to be late or absent.",
    "c1": "Employees are entitled to 20 days of paid annual leave per calendar year, in addition to public holidays. Leave requests must be submitted to the reporting manager at least one week in advance. Sick leave may require a medical certificate for absences exceeding three consecutive working days.",
    "c2": "All employees must maintain the confidentiality of company information, including customer data, financial records, business strategies, and internal communications. Sharing confidential information with unauthorized individuals is strictly prohibited, even after employment ends.",
    "c3": "The company follows a hybrid work policy, allowing employees to work remotely up to two days per week with manager approval. Employees working remotely must remain available during core business hours and ensure they have a secure internet connection.",
    "c4": "Employees must use company-provided laptops, software, and other resources primarily for business purposes. Installing unauthorized software, sharing login credentials, or accessing inappropriate websites using company devices is prohibited.",
}

chunkIds = list(chunks.keys())
chunkContents = list(chunks.values())
embeddings = []
for c in chunkContents:
    embeddings.append(create_embedding(c))

# Storing the embeddings generated from the chunks in vectorDB, it generates embedding itself.
# By default, it is using all-MiniLM-L6-v2 model.
# upsert helps in case documents, embeddings gets updated
collection.upsert(ids=chunkIds, documents=chunkContents, embeddings=embeddings)

# Semantic seraching the user query.
userQuery = "What are the leave policies ?"
query_embedding = create_embedding(userQuery)
results = collection.query(query_embeddings=[query_embedding], n_results=2)

print(results["documents"])
