import os

import numpy

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    # Not needed as we are hitting local LLM, adding dummy key since OpenAI has made it non null
    api_key=os.getenv("API_KEY"),
)


def get_embedding(text):
    response = client.embeddings.create(input=text, model=os.getenv("EMBED_MODEL"))
    return response.data[0].embedding


def cosine_similarity(embedding1, embedding2):
    dot_prod = numpy.dot(embedding1, embedding2)
    embedding1_len = numpy.linalg.norm(embedding1)
    embedding2_len = numpy.linalg.norm(embedding2)

    return dot_prod / (embedding1_len * embedding2_len)


text1 = "Python is a programming language"
text2 = "Numpy is a library used for heavy mathematical calculations"
text3 = "I love pizza"
text4 = "I hate pizza"

embedding1 = get_embedding(text1)
embedding2 = get_embedding(text2)
embedding3 = get_embedding(text3)
embedding4 = get_embedding(text4)

print(cosine_similarity(embedding1, embedding2))
print(cosine_similarity(embedding1, embedding3))
print(cosine_similarity(embedding2, embedding3))
print(cosine_similarity(embedding3, embedding4))
