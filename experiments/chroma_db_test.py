import chromadb
from dotenv import load_dotenv

load_dotenv()

userQuery = "What are the leave policies ?"

client = chromadb.Client()
collection = client.create_collection(name="NOVATECH_SOLUTIONS")

chunks = {
    "c0": "Employees are expected to work 40 hours per week, typically from Monday to Friday. Standard working hours are 9:00 AM to 6:00 PM, with a one-hour lunch break. Employees must inform their managers if they expect to be late or absent.",
    "c1": "Employees are entitled to 20 days of paid annual leave per calendar year, in addition to public holidays. Leave requests must be submitted to the reporting manager at least one week in advance. Sick leave may require a medical certificate for absences exceeding three consecutive working days.",
    "c2": "All employees must maintain the confidentiality of company information, including customer data, financial records, business strategies, and internal communications. Sharing confidential information with unauthorized individuals is strictly prohibited, even after employment ends.",
    "c3": "The company follows a hybrid work policy, allowing employees to work remotely up to two days per week with manager approval. Employees working remotely must remain available during core business hours and ensure they have a secure internet connection.",
    "c4": "Employees must use company-provided laptops, software, and other resources primarily for business purposes. Installing unauthorized software, sharing login credentials, or accessing inappropriate websites using company devices is prohibited.",
}

chunkIds = list(chunks.keys())
chunkContents = list(chunks.values())

# Storing the embeddings generated from the chunks in vectorDB, it generates embedding itself.
# By default, it is using all-MiniLM-L6-v2 model.
collection.add(ids=chunkIds, documents=chunkContents)
# Semantic seraching the user query.
results = collection.query(query_texts=[userQuery], n_results=2)
print(results["documents"])
