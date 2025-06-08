from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer


class ContextAwareChunker:
    """A class that implements context-aware chunking strategies."""
    
    def __init__(self, strategy="semantic_units", chunk_size=1000, chunk_overlap=200, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the chunker with the specified strategy.
        
        Args:
            strategy: The chunking strategy to use. Options:
                - "semantic_units": Split by paragraphs, headers, and lists
                - "embedding_similarity": Group similar sentences together
                - "topic_segmentation": Split by topic shifts
                - "default": Use basic RecursiveCharacterTextSplitter
            chunk_size: The target size of each chunk
            chunk_overlap: The overlap between chunks
            model_name: The sentence transformer model for embedding-based strategies
        """
        self.strategy = strategy
        
        # Ensure chunk_size and chunk_overlap have valid default values
        self.chunk_size = 1000 if chunk_size is None else chunk_size
        self.chunk_overlap = 200 if chunk_overlap is None else chunk_overlap
        
        # Initialize the basic splitter as fallback
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Load model for embedding-based strategies if needed
        if strategy == "embedding_similarity":
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logging.error(f"Failed to load sentence transformer model: {e}")
                self.strategy = "default"
                
        logging.info(f"ContextAwareChunker initialized with strategy: {strategy}")
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk the input text based on the selected strategy.
        
        Args:
            text: The input text to chunk
            
        Returns:
            A list of dictionaries containing the text chunks and their metadata
        """
        if self.strategy == "semantic_units":
            return self._chunk_by_semantic_units(text)
        elif self.strategy == "embedding_similarity":
            return self._chunk_by_embedding_similarity(text)
        elif self.strategy == "topic_segmentation":
            return self._chunk_by_topic_segmentation(text)
        else:
            # Default to basic chunking
            chunks = self.default_splitter.create_documents([text])
            return [{"text": chunk.page_content, "metadata": {"strategy": "default"}} for chunk in chunks]
    
    def _chunk_by_semantic_units(self, text: str) -> List[Dict[str, Any]]:
        """Split text by paragraphs, headers, and list items."""
        chunks = []
        current_chunk = ""
        current_section = "Unknown"
        current_metadata = {"strategy": "semantic_units", "section": current_section}
        
        # Regular expressions for detecting headers and structural elements
        header_pattern = re.compile(r'^#{1,6}\s+(.+)$|^(.+)\n[=\-]{3,}$', re.MULTILINE)
        list_pattern = re.compile(r'^\s*[-*•]\s+|^\s*\d+\.\s+', re.MULTILINE)
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if paragraph is a header
            header_match = header_pattern.match(paragraph)
            if header_match:
                # If we have content in the current chunk, add it to chunks
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(), 
                        "metadata": current_metadata.copy()
                    })
                    current_chunk = ""
                    
                # Update the section title
                current_section = header_match.group(1) if header_match.group(1) else header_match.group(2)
                current_metadata = {
                    "strategy": "semantic_units", 
                    "section": current_section,
                    "is_header": True,
                    "paragraph_index": i
                }
                current_chunk = paragraph
            
            # Check if this would make the chunk too large
            elif len(current_chunk) + len(paragraph) + 1 > self.chunk_size:
                # If the current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(), 
                        "metadata": current_metadata.copy()
                    })
                
                # Start a new chunk
                current_chunk = paragraph
                current_metadata = {
                    "strategy": "semantic_units", 
                    "section": current_section,
                    "paragraph_index": i
                }
                
                # Check if paragraph is a list item
                if list_pattern.match(paragraph):
                    current_metadata["content_type"] = "list_item"
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
                
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(), 
                "metadata": current_metadata.copy()
            })
            
        return chunks
    
    def _chunk_by_embedding_similarity(self, text: str) -> List[Dict[str, Any]]:
        """Group sentences with similar embeddings together."""
        from sklearn.cluster import AgglomerativeClustering
        
        # First split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            return [{"text": text, "metadata": {"strategy": "embedding_similarity", "cluster": 0}}]
            
        # Compute embeddings for sentences
        try:
            embeddings = self.model.encode(sentences)
            
            # Determine optimal number of clusters (simplified approach)
            n_clusters = max(1, min(len(sentences) // 5, int(len(text) / self.chunk_size)))
            
            # Cluster sentences based on embedding similarity
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
            clusters = clustering.fit_predict(embeddings)
            
            # Group sentences by cluster
            grouped_chunks = {}
            for i, (sentence, cluster) in enumerate(zip(sentences, clusters)):
                if cluster not in grouped_chunks:
                    grouped_chunks[cluster] = {
                        "text": sentence,
                        "metadata": {
                            "strategy": "embedding_similarity",
                            "cluster": int(cluster),
                            "first_sentence_index": i
                        }
                    }
                else:
                    grouped_chunks[cluster]["text"] += " " + sentence
            
            # Sort chunks by the order they appeared in the text
            chunks = sorted(grouped_chunks.values(), key=lambda x: x["metadata"]["first_sentence_index"])
            return chunks
            
        except Exception as e:
            logging.error(f"Error in embedding-based chunking: {e}")
            # Fallback to default chunking
            return self._chunk_by_semantic_units(text)
    
    def _chunk_by_topic_segmentation(self, text: str) -> List[Dict[str, Any]]:
        """Split text by topic shifts using TextTiling algorithm."""
        try:
            from nltk.tokenize.texttiling import TextTilingTokenizer
            
            # Initialize TextTiling tokenizer
            tt = TextTilingTokenizer(w=20, k=10)
            segments = tt.tokenize(text)
            
            # Create chunks from the segments with metadata
            chunks = []
            for i, segment in enumerate(segments):
                if segment.strip():  # Skip empty segments
                    chunks.append({
                        "text": segment.strip(),
                        "metadata": {
                            "strategy": "topic_segmentation",
                            "segment_index": i
                        }
                    })
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error in topic-based chunking: {e}. Using semantic units instead.")
            # Fallback to semantic units
            return self._chunk_by_semantic_units(text)


# Function for backward compatibility with the existing interface
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Legacy function that uses the default RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks