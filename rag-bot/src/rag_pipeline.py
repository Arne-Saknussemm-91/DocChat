from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import requests
import json

class RAGPipeline:
    def __init__(self, vector_store, model_name="mistral:latest"):
        self.vector_store = vector_store
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api"
        self.test_ollama_connection()

    def test_ollama_connection(self):
        try:
            response = requests.get(f"{self.api_base}/version")
            if response.status_code == 200:
                print("Successfully connected to Ollama API")
            else:
                print(f"Warning: Ollama API returned status code {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama API: {str(e)}")

    def generate_answer(self, query):
        try:
            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=3)
            
            # Prepare context
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt for Mistral
            prompt = f"""<s>[INST]Using the following context, please answer the question.
            Keep your response focused and relevant to the context provided.

            Context: {context}

            Question: {query}[/INST]</s>"""
            
            # Call Ollama API
            response = requests.post(
                f"{self.api_base}/generate",
                headers={'Content-Type': 'application/json'},
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                print(f"API Error Details: {response.text}")
                return f"Error: Failed to generate response. Status code: {response.status_code}"
                
        except Exception as e:
            return f"Error generating answer: {str(e)}"