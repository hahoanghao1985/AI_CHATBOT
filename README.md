# AI Support Chatbot

An advanced AI-powered support chatbot with enhanced RAG capabilities, featuring voice interaction, multi-language support, and comprehensive document processing. Built with FastAPI backend and a modern web frontend with real-time voice controls.

## Features

### 🎤 Voice & Multi-Language Support
- **Voice Recognition**: Hands-free document queries with Web Speech API
  - Language selector dropdown (English, Tiếng Việt, French, Spanish)
  - Real-time voice status feedback
  - Voice commands for sending messages ("send", "gửi đi", "envoyer", "enviar")
- **Text-to-Speech**: Listen to AI responses with natural voices
  - Individual speaker controls for each AI message
  - Stop/pause functionality during playback
  - Automatic language detection for appropriate voice selection
  - Support for Vietnamese, English, French, and Spanish voices
- **Multi-Language Interface**: Seamless support for multiple languages
  - Language-specific voice command recognition
  - Smart text language detection for speech synthesis
  - Consistent experience across all supported languages

### 🌐 Web Interface
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
  - **Voice Controls**: Integrated voice input and audio playback

### 🚀 Enhanced Backend (FastAPI)
- Advanced RAG pipeline with configurable parameters
- Multiple embedding model support (OpenAI, Cohere)
- Document processing with metadata extraction for source attribution
- Re-ranking capabilities (Cohere Rerank, LLM-based)
- Context compression for optimal performance
- ChromaDB vector storage with rich metadata
- Support for multiple file formats and web URLs
- Static file serving for the web interface with voice capabilities

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
   - **Voice Interaction**:
     - Select your preferred language from the dropdown (English, Tiếng Việt, French, Spanish)
     - Click the microphone button to start voice input
     - Say your question in the selected language
     - Use voice commands to send messages: "send", "gửi đi", "envoyer", "enviar"
     - Click the speaker (🔊) button next to any AI response to hear it spoken aloud
     - Use the stop (⏹️) button to pause speech playback
   - Ask questions and receive answers with detailed source references
   - View document titles, page numbers, and file paths for each answer

## Advanced Features

### 📖 Source References
Every AI response includes detailed source information:
- **Document Title**: Original document name or extracted title
- **File Path**: Location of the source document
- **Page Number**: Specific page reference (for PDFs)
- **File Type**: Document format indicator
- **Web Links**: Clickable links for web-based sources

### 🎯 Configurable RAG Pipeline
- **Embedding Models**: Choose between OpenAI and Cohere embeddings
- **Re-ranking**: Improve relevance with Cohere Rerank or LLM-based re-ranking
- **Context Compression**: Automatically compress large contexts for optimal performance
- **Chunk Control**: Fine-tune retrieval with configurable chunk counts

### 🎵 Voice & Audio Features
- **Speech Recognition**: Browser-based voice input using Web Speech API
  - Supports English, Vietnamese, French, and Spanish
  - Real-time language switching via dropdown selection
  - Voice status feedback with interim results display
- **Text-to-Speech**: Natural voice synthesis for AI responses
  - Automatic language detection for appropriate voice selection
  - Individual playback controls for each message
  - Pause/stop functionality during speech
- **Voice Commands**: Hands-free message sending
  - English: "send", "send it"
  - Vietnamese: "gửi", "gửi đi", "gui", "gui di"
  - French: "envoyer", "envoi"
  - Spanish: "enviar", "envía"

### 🌍 Multi-Language Support
- **Interface Languages**: Full support for multiple languages
- **Voice Recognition**: Language-specific speech recognition
- **Smart Detection**: Automatic language detection for text-to-speech
- **Consistent Experience**: Unified interface across all supported languages

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
- **Modern web browser** with Web Speech API support:
  - Chrome 25+ (recommended for best voice support)
  - Firefox 44+
  - Safari 14.1+
  - Edge 79+
- **Microphone access** for voice input functionality
- **Audio output** for text-to-speech playback
- Windows/macOS/Linux support

### Browser Compatibility Notes
- **Voice Recognition**: Requires HTTPS in production or localhost for development
- **Text-to-Speech**: Supported in all modern browsers
- **Language Support**: Voice quality may vary by browser and operating system
- **Permissions**: Browser will request microphone access on first voice input attempt

## 🎤 Voice Features Guide

### Getting Started with Voice
1. **Enable Microphone**: Grant microphone permission when prompted
2. **Select Language**: Choose your preferred language from the dropdown
3. **Start Speaking**: Click the microphone button and speak your question
4. **Voice Commands**: Say send commands in your language to auto-submit messages
5. **Listen to Responses**: Click the speaker icon (🔊) next to any AI response

### Voice Tips & Best Practices
- **Clear Speech**: Speak clearly and at a moderate pace for best recognition
- **Quiet Environment**: Use in a quiet environment for optimal voice recognition
- **Language Consistency**: Select the language you plan to speak before starting
- **Voice Commands**: Remember the send commands:
  - 🇺🇸 English: "send" or "send it"
  - 🇻🇳 Vietnamese: "gửi đi" or "gửi"
  - 🇫🇷 French: "envoyer" or "envoi"
  - 🇪🇸 Spanish: "enviar" or "envía"
- **Audio Playback**: Use headphones to prevent audio feedback during voice input

### Troubleshooting Voice Issues
- **No Microphone Access**: Check browser permissions and system microphone settings
- **Poor Recognition**: Try speaking more clearly or switching to a quieter environment
- **Wrong Language**: Ensure the correct language is selected in the dropdown
- **No Audio Playback**: Check system volume and browser audio permissions
- **Voice Commands Not Working**: Make sure to say the exact command words listed above

## How to run:
```bash
# Quick start
python backend/app.py

# Alternative with uvicorn
uvicorn backend.app:app --reload --port 8000

# Then open http://localhost:8000 in your browser
# Grant microphone permissions for voice features
```