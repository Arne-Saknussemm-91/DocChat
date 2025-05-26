from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
import os

class VectorStore:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        # Set up persistent directory
        persist_directory = os.path.join(os.getcwd(), "db")
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings and vector store
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def similarity_search(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)
    
    def index_embeddings(self, chunks):
        try:
            if chunks:
                self.vectorstore.add_texts(texts=chunks)
            return True
        except Exception as e:
            print(f"Error indexing embeddings: {str(e)}")
            return False