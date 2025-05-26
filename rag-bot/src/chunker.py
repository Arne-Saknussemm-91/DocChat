from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=300, overlap=60):
    """
    Splits the input text into smaller chunks with specified overlap.
    
    Parameters:
    - text (str): The text to be chunked.
    - chunk_size (int): The size of each chunk in characters.
    - overlap (int): The number of overlapping characters between chunks.
    
    Returns:
    - List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks