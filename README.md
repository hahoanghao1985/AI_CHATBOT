# AI Support Chatbot

An advanced AI-powered support chatbot with enhanced RAG capabilities, featuring a FastAPI backend and a modern web frontend with comprehensive document source references.

## Features

### Web Interface
- **Upload Tab**: Multiple document upload with drag-and-drop support
  - Support for PDF, DOCX, DOC, TXT, HTML files and URLs
  - Embedding model selection (OpenAI text-embedding-3-small/large, Cohere V3)
  - Real-time upload progress and status
- **Chat Tab**: Advanced query interface with configurable RAG parameters
  - AI model selection (GPT-3.5, GPT-4, GPT-4 Turbo)
  - Configurable chunk count (3, 5, 7, 10, 20, 30)
  - Re-ranking models (Cohere Rerank, LLM-based, None)
  - Context compression for large documents
  - **Source References**: Detailed citations showing document titles, page numbers, and file paths

### Enhanced Backend (FastAPI)
- Advanced RAG pipeline with configurable parameters
- Multiple embedding model support (OpenAI, Cohere)
- Document processing with metadata extraction for source attribution
- Re-ranking capabilities (Cohere Rerank, LLM-based)
- Context compression for optimal performance
- ChromaDB vector storage with rich metadata
- Support for multiple file formats and web URLs

## Structure
```
ai-support-chatbot/
├── backend/                # FastAPI backend with enhanced RAG
│   ├── app.py             # Main API endpoints with source references
│   ├── rag_pipeline.py    # Advanced RAG logic with re-ranking & compression
│   ├── ingestion.py       # Multi-format document processing & metadata extraction
│   └── requirements.txt   # Python dependencies
├── frontend/              # Modern web interface
│   └── web-app.html      # Complete web application with advanced controls
├── uploads/               # Document storage
├── vector_db/            # ChromaDB vector storage
└── README.md
```

## Setup and Installation

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

2. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here  # Optional, for Cohere re-ranking
   ```

4. Start the backend server:
   ```bash
   python backend/app.py
   ```
   Or using uvicorn:
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```

### Frontend Access
The web interface is served directly by the FastAPI backend at `http://localhost:8000`
- No additional frontend setup required
- Access the application through any modern web browser

## Usage

1. **Start the Application**: Run `python backend/app.py` to start the server on `http://localhost:8000`
2. **Access Web Interface**: Open your browser and navigate to `http://localhost:8000`
3. **Upload Documents**: Use the "Upload Documents" tab to:
   - Select embedding model (text-embedding-3-small, text-embedding-3-large, Cohere V3)
   - Upload PDF, DOCX, DOC, TXT, HTML files or provide URLs
   - View upload progress and processing status
4. **Query Documents**: Switch to "Chat & Query" tab to:
   - Configure RAG parameters:
     - AI model selection (GPT-3.5, GPT-4, etc.)
     - Embedding model for retrieval
     - Number of chunks to retrieve (3-30)
     - Re-ranking method (None, Cohere, LLM-based)
     - Context compression toggle
   - Ask questions and receive answers with detailed source references
   - View document titles, page numbers, and file paths for each answer

## Advanced Features

### Source References
Every AI response includes detailed source information:
- **Document Title**: Original document name or extracted title
- **File Path**: Location of the source document
- **Page Number**: Specific page reference (for PDFs)
- **File Type**: Document format indicator
- **Web Links**: Clickable links for web-based sources

### Configurable RAG Pipeline
- **Embedding Models**: Choose between OpenAI and Cohere embeddings
- **Re-ranking**: Improve relevance with Cohere Rerank or LLM-based re-ranking
- **Context Compression**: Automatically compress large contexts for optimal performance
- **Chunk Control**: Fine-tune retrieval with configurable chunk counts

## Development

### API Endpoints
- `POST /upload/`: Upload multiple documents with embedding model selection
- `POST /chat/`: Send queries with configurable RAG parameters
- `DELETE /clear-database/`: Clear all stored documents and embeddings
- `GET /database-status/`: Check database status and document count
- `GET /`: Serve the web interface

### Dependencies
Key Python packages:
- `fastapi`: Web framework and API
- `langchain`: RAG pipeline and document processing
- `langchain-openai`: OpenAI integration
- `langchain-cohere`: Cohere embedding and re-ranking
- `chromadb`: Vector database
- `pypdf2`, `python-docx`: Document processing
- `beautifulsoup4`: HTML processing

## Requirements

- Python 3.8+
- OpenAI API key (required)
- Cohere API key (optional, for enhanced re-ranking)
- Modern web browser
- Windows/macOS/Linux support

## How to run:
```bash
# Quick start
python backend/app.py

# Alternative with uvicorn
uvicorn backend.app:app --reload --port 8000

# Then open http://localhost:8000 in your browser
```