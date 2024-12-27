# Document Q&A RAG Assistant

An intelligent document question-answering system built with Streamlit that uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from uploaded documents.

## Features

- **Document Processing**: Support for PDF and TXT file uploads
- **Smart Text Chunking**: Implements recursive character text splitting for optimal context preservation
- **Vector Storage**: Uses ChromaDB for efficient document embedding storage
- **Advanced Retrieval**: 
  - Employs Ollama embeddings for semantic search
  - Cross-encoder reranking for improved relevance
- **LLM Integration**: Powered by llama3.2 for generating natural language responses
- **Interactive UI**: Clean Streamlit interface for easy document upload and querying

## Prerequisites

- Python 3.8+
- Ollama running locally
- Required models:
  - nomic-embed-text (for embeddings)
  - llama3.2:3b (for text generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pankajjoshi11/Document-Q-A-RAG-assistant.git
cd document-qa-rag-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama:
```bash
# Follow Ollama installation instructions from https://ollama.ai
ollama run llama3.2:3b
ollama run nomic-embed-text
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload a document (PDF or TXT) using the sidebar
3. Click "Process" to analyze the document
4. Enter your question in the text area
5. Click "Ask" to get your answer

## Technical Architecture

1. **Document Processing Pipeline**:
   - Document upload and parsing using PyMuPDF
   - Text splitting using RecursiveCharacterTextSplitter
   - Vector embedding generation using Ollama

2. **Query Processing**:
   - Vector similarity search in ChromaDB
   - Cross-encoder reranking for relevance optimization
   - Context-aware response generation using LLM

3. **Storage**:
   - Persistent vector storage using ChromaDB
   - Document metadata preservation

## Dependencies

- streamlit
- langchain
- chromadb
- sentence-transformers
- ollama
- PyMuPDF

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
