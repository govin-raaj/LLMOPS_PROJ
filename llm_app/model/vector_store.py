import chromadb
from langchain.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, path):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


        self.vector_store = Chroma(
            persist_directory=path,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)
        
    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)
        