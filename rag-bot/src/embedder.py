from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, chunks):
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks (list of str): List of text chunks to be embedded.

        Returns:
            list: List of embeddings corresponding to the input chunks.
        """
        embeddings = self.model.encode(chunks, convert_to_tensor=True)
        return embeddings.tolist()  # Convert to list for easier handling

    def save_embeddings(self, embeddings, file_path):
        """
        Save embeddings to a file.

        Args:
            embeddings (list): List of embeddings to save.
            file_path (str): Path to the file where embeddings will be saved.
        """
        import json
        with open(file_path, 'w') as f:
            json.dump(embeddings, f)

    def load_embeddings(self, file_path):
        """
        Load embeddings from a file.

        Args:
            file_path (str): Path to the file from which embeddings will be loaded.

        Returns:
            list: List of loaded embeddings.
        """
        import json
        with open(file_path, 'r') as f:
            embeddings = json.load(f)
        return embeddings