from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
import os
from pymilvus import connections
from dotenv import load_dotenv
from uuid import uuid4
from langchain_core.documents import Document

load_dotenv()

# Milvus connection details
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "langchain_milvus_demo"

# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY")
)

print("loaded embeddings")

connection_args = {
    "host": "localhost",
    "port": "19530",
}

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

vector_db = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name="langchain_collection",
)

document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_db.add_documents(documents=documents, ids=uuids)
