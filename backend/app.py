import shutil
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ingestion import process_document, clear_vector_database, simple_clear_vector_database, get_database_status, inspect_database_tables
from rag_pipeline import get_answer
from dotenv import load_dotenv

#Load OpenAI API key from .env file
load_dotenv()

print("API Key loaded:", os.getenv("OPENAI_API_KEY") is not None)

app = FastAPI()

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.html', '.htm'}

class URLRequest(BaseModel):
    url: str
    embedding_model: str = "text-embedding-3-small"

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...), embedding_model: str = Form("text-embedding-3-small")):
    import sys
    results = []
    total_chunks = 0
    
    print(f"=== UPLOAD DEBUG: Received {len(files)} files with embedding model: {embedding_model} ===", flush=True)
    sys.stdout.flush()
    
    for file in files:
        if file.filename:
            print(f"Processing file: {file.filename}", flush=True)
            sys.stdout.flush()
            
            # Check file extension
            file_ext = os.path.splitext(file.filename)[1].lower()
            print(f"File extension: {file_ext}", flush=True)
            sys.stdout.flush()
            
            if file_ext not in SUPPORTED_EXTENSIONS:
                print(f"Unsupported file type: {file_ext}", flush=True)
                sys.stdout.flush()
                results.append({
                    "filename": file.filename, 
                    "status": "error", 
                    "message": f"Unsupported file type: {file_ext}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
                })
                continue
            
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            print(f"Saving file to: {file_path}", flush=True)
            sys.stdout.flush()
            
            try:
                # Save file first
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                print(f"File saved successfully: {file_path}", flush=True)
                sys.stdout.flush()
                
                print(f"Starting document processing...", flush=True)
                sys.stdout.flush()
                
                # For Windows compatibility, use threading timeout instead of signal
                import threading
                import time
                
                result = {"chunks_count": 0, "error": None}
                
                def process_with_timeout():
                    try:
                        result["chunks_count"] = process_document(file_path, embedding_model)
                    except Exception as e:
                        result["error"] = e
                
                # Start processing in a separate thread
                process_thread = threading.Thread(target=process_with_timeout)
                process_thread.daemon = True
                process_thread.start()
                
                # Wait for completion with timeout
                process_thread.join(timeout=120)  # 2 minute timeout
                
                if process_thread.is_alive():
                    print(f"TIMEOUT ERROR: Document processing timed out after 2 minutes", flush=True)
                    sys.stdout.flush()
                    results.append({
                        "filename": file.filename, 
                        "status": "error",
                        "message": "Processing timeout after 2 minutes"
                    })
                elif result["error"]:
                    raise result["error"]
                else:
                    chunks_count = result["chunks_count"]
                    print(f"Document processing completed. Chunks added: {chunks_count}", flush=True)
                    sys.stdout.flush()
                    
                    total_chunks += chunks_count
                    results.append({
                        "filename": file.filename, 
                        "status": "success",
                        "chunks_added": chunks_count,
                        "embedding_model": embedding_model
                    })
                    
            except Exception as e:
                print(f"ERROR processing file {file.filename}: {str(e)}", flush=True)
                print(f"Exception type: {type(e).__name__}", flush=True)
                sys.stdout.flush()
                import traceback
                traceback.print_exc()
                
                results.append({
                    "filename": file.filename, 
                    "status": "error",
                    "message": str(e)
                })
    
    print(f"=== UPLOAD DEBUG: Completed processing {len(files)} files ===", flush=True)
    sys.stdout.flush()
    
    return {
        "status": "completed", 
        "files_processed": len([r for r in results if r.get("status") == "success"]), 
        "total_chunks": total_chunks, 
        "results": results
    }

@app.post("/upload-url/")
async def upload_url(request: URLRequest):
    try:
        chunks_count = process_document(request.url, request.embedding_model)
        return {
            "status": "success", 
            "url": request.url,
            "chunks_added": chunks_count,
            "embedding_model": request.embedding_model
        }
    except Exception as e:
        return {
            "status": "error", 
            "url": request.url,
            "message": str(e)
        }

@app.post("/chat/")
async def chat(
    query: str = Form(...), 
    model: str = Form(default="gpt-3.5-turbo"), 
    embedding_model: str = Form(default="text-embedding-3-small"),
    chunk_count: int = Form(default=3),
    reranker_type: str = Form(default="none"),
    use_compression: bool = Form(default=True)
):
    response = get_answer(
        query, 
        model, 
        embedding_model, 
        chunk_count, 
        reranker_type, 
        use_compression
    )
    
    # response is now a dictionary with answer, sources, etc.
    return {
        "answer": response["answer"], 
        "sources": response.get("sources", []),
        "model_used": model, 
        "embedding_model_used": embedding_model,
        "chunk_count": chunk_count,
        "chunks_used": response.get("chunks_used", chunk_count),
        "reranker_type": reranker_type,
        "compression_used": response.get("compression_used", use_compression),
        "language_detected": response.get("language_detected", "english")
    }

@app.delete("/clear-database/")
async def clear_database():
    """Clear all data from the vector database and uploaded files"""
    try:
        # Try the comprehensive clear first
        result = clear_vector_database()
        
        # If the comprehensive clear had issues but collections were cleared, that's still success
        if result.get("success") and result.get("vector_db_cleared"):
            message = "Successfully cleared: "
            cleared_items = []
            if result["vector_db_cleared"]:
                cleared_items.append("vector database")
            if result.get("uploads_cleared"):
                cleared_items.append("uploaded files")
            
            if cleared_items:
                message += " and ".join(cleared_items)
            else:
                message += "database (was already empty)"
                
            return {
                "status": "success",
                "message": message,
                "details": result
            }
        else:
            # If comprehensive clear failed, try simple clear
            print("Comprehensive clear failed, trying simple clear...")
            simple_result = simple_clear_vector_database()
            
            if simple_result.get("success"):
                return {
                    "status": "success", 
                    "message": "Successfully cleared vector database using simple method",
                    "details": simple_result
                }
            else:
                return {
                    "status": "error",
                    "message": f"Both clearing methods failed. Comprehensive: {result.get('error', 'Unknown error')}. Simple: {simple_result.get('error', 'Unknown error')}",
                    "details": {"comprehensive": result, "simple": simple_result}
                }
                
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error clearing data: {str(e)}"
        }

@app.delete("/simple-clear-database/")
async def simple_clear_database():
    """Simple alternative database clearing method"""
    try:
        result = simple_clear_vector_database()
        if result.get("success"):
            return {
                "status": "success",
                "message": "Successfully cleared vector database using simple method",
                "details": result
            }
        else:
            return {
                "status": "error", 
                "message": f"Simple clear failed: {result.get('error', 'Unknown error')}",
                "details": result
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in simple clear: {str(e)}"
        }

@app.get("/database-status/")
async def get_database_status_endpoint():
    """Get current status of the vector database and uploaded files"""
    try:
        status = get_database_status()
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting database status: {str(e)}"
        }

@app.get("/inspect-database/")
async def inspect_database():
    """Inspect the SQLite database tables and their contents"""
    try:
        result = inspect_database_tables()
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error inspecting database: {str(e)}"
        }

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download uploaded files"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Security check: ensure file is in upload directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(UPLOAD_DIR)):
            return {
                "status": "error",
                "message": "Invalid file path"
            }
        
        if not os.path.exists(file_path):
            return {
                "status": "error", 
                "message": "File not found"
            }
        
        # Get file extension to set proper media type
        file_ext = os.path.splitext(filename)[1].lower()
        media_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html'
        }
        
        media_type = media_type_map.get(file_ext, 'application/octet-stream')
        
        return FileResponse(
            path=file_path, 
            filename=filename,
            media_type=media_type
        )
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error downloading file: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
