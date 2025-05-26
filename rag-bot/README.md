# RAG Bot

## Overview
The RAG Bot is a Retrieval-Augmented Generation (RAG) system designed to analyze PDF documents of up to 100 pages. It extracts text, chunks it for context preservation, generates embeddings, and utilizes a generative model (LLaMA-3) to provide concise answers based on user queries.

## Project Structure
```
rag-bot
├── src
│   ├── main.py          # Entry point of the application
│   ├── pdf_extractor.py # Extracts text from PDF documents
│   ├── chunker.py      # Splits text into manageable chunks
│   ├── embedder.py      # Converts text chunks into embeddings
│   ├── vector_store.py  # Manages storage and retrieval of embeddings
│   ├── rag_pipeline.py   # Integrates retrieval and generative components
│   └── utils.py        # Utility functions for various tasks
├── requirements.txt     # Lists project dependencies
└── README.md            # Documentation for the project
```

## Installation
To set up the RAG Bot, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd rag-bot
pip install -r requirements.txt
```

## Usage
1. Place your PDF document (up to 100 pages) in the appropriate directory.
2. Run the application:

```bash
python src/main.py
```

3. Follow the prompts to ask questions about the content of the PDF.

## Dependencies
The project requires the following Python libraries:
- pdfplumber
- sentence-transformers
- langchain
- ollama (for LLaMA-3)
- other necessary packages

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.