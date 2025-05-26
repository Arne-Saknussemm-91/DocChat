import os
import argparse
from pdf_extractor import extract_text_from_pdf
from chunker import chunk_text
from embedder import Embedder
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to the PDF file')
    args = parser.parse_args()
    
    # Get PDF path from args or input
    pdf_file_path = args.file or input("Please enter the path to the PDF file (up to 100 pages): ")
    
    # Convert relative path to absolute path and normalize
    pdf_file_path = os.path.abspath(os.path.normpath(pdf_file_path))
    
    # Debug output
    print(f"Looking for file at: {pdf_file_path}")
    print(f"File exists: {os.path.exists(pdf_file_path)}")
    print(f"Current working directory: {os.getcwd()}")
    
    if not os.path.exists(pdf_file_path):
        print(f"Error: File not found at {pdf_file_path}")
        return
        
    print(f"Processing PDF file: {pdf_file_path}")
    
    # Step 2: Extract text from the PDF
    print("Extracting text from the PDF...")
    extracted_text = extract_text_from_pdf(pdf_file_path)

    # Step 3: Chunk the extracted text
    print("Chunking the extracted text...")
    text_chunks = chunk_text(extracted_text)

    # Step 4: Generate embeddings for the text chunks
    print("Generating embeddings for the text chunks...")
    vector_store = VectorStore()
    vector_store.index_embeddings(text_chunks)

    # Step 5: Set up the RAG pipeline
    rag_pipeline = RAGPipeline(vector_store)

    # Step 6: Handle user queries
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        print("\nGenerating answer...")
        answer = rag_pipeline.generate_answer(user_query)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()