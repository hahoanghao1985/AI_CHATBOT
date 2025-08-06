# Document processing & embedding logic
# Support for multiple file types: PDF, DOC, DOCX, TXT, HTML

import fitz  # PyMuPDF for PDFs
import os
import requests
from typing import Union
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from docx import Document  # python-docx for Word documents
from bs4 import BeautifulSoup  # BeautifulSoup for HTML parsing
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHROMA_DB_DIR = "vector_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

def get_embeddings(model: str = "text-embedding-3-small"):
    """Get embeddings with specified model"""
    if model.startswith("text-embedding"):
        return OpenAIEmbeddings(model=model)
    elif model == "cohere-v3":
        # Note: You would need to install cohere and configure API key for this
        # from langchain_cohere import CohereEmbeddings
        # return CohereEmbeddings(model="embed-english-v3.0")
        # For now, fallback to OpenAI
        print(f"Cohere embeddings not implemented, falling back to text-embedding-3-small")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Default fallback
        return OpenAIEmbeddings(model="text-embedding-3-small")

def extract_text_from_pdf(file_path: str) -> tuple[str, list[dict]]:
    """Extract text from PDF files using PyMuPDF and return text with page metadata"""
    doc = fitz.open(file_path)
    text = ""
    page_metadata = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        text += page_text
        if page_text.strip():  # Only add metadata for pages with content
            page_metadata.append({
                "page_number": page_num + 1,  # 1-based page numbering
                "char_start": len(text) - len(page_text),
                "char_end": len(text),
                "page_text_length": len(page_text)
            })
    
    doc.close()
    return text, page_metadata

def extract_text_from_docx(file_path: str) -> tuple[str, list[dict]]:
    """Extract text from DOCX files using python-docx and return text with estimated page metadata"""
    try:
        print(f"Processing DOCX file: {file_path}")
        doc = Document(file_path)
        text = ""
        page_metadata = []
        
        # Estimate page breaks based on text length
        # Assuming ~500 words per page (approximately 3000 characters including spaces)
        chars_per_page = 3000
        
        for para_num, paragraph in enumerate(doc.paragraphs, 1):
            para_text = paragraph.text + "\n"
            text += para_text
            if para_text.strip():  # Only add metadata for paragraphs with content
                # Calculate estimated page number based on character position
                char_start = len(text) - len(para_text)
                estimated_page = max(1, (char_start // chars_per_page) + 1)
                
                page_metadata.append({
                    "page_number": estimated_page,
                    "paragraph_number": para_num,
                    "char_start": char_start,
                    "char_end": len(text),
                    "paragraph_text_length": len(para_text),
                    "estimated_page": True  # Flag to indicate this is estimated
                })
        
        print(f"Successfully extracted {len(text)} characters from DOCX")
        return text, page_metadata
        
    except Exception as e:
        print(f"Error processing DOCX file {file_path}: {str(e)}")
        error_text = f"Error processing DOCX file: {str(e)}"
        return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True}]

def extract_text_from_doc(file_path: str) -> tuple[str, list[dict]]:
    """Extract text from DOC files (basic support) and return text with basic metadata"""
    try:
        # Try to read as binary and extract readable text
        with open(file_path, 'rb') as file:
            content = file.read()
            # Simple extraction - may not work perfectly for all DOC files
            text = content.decode('utf-8', errors='ignore')
            # Clean up the text
            text = ''.join(char for char in text if char.isprintable() or char.isspace())
            
            # Basic metadata for DOC files
            page_metadata = [{
                "page_number": 1,
                "char_start": 0,
                "char_end": len(text),
                "note": "DOC file - exact page numbers not available"
            }]
            
            return text, page_metadata
    except Exception as e:
        print(f"Error reading DOC file {file_path}: {e}")
        error_text = f"Error reading DOC file: {e}"
        return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True}]

def extract_text_from_txt(file_path: str) -> tuple[str, list[dict]]:
    """Extract text from TXT files and return text with line metadata"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            text = file.read()
    
    # Basic metadata for TXT files
    page_metadata = [{
        "page_number": 1,
        "char_start": 0,
        "char_end": len(text),
        "line_count": len(text.split('\n'))
    }]
    
    return text, page_metadata

def extract_text_from_html_file(file_path: str) -> tuple[str, list[dict]]:
    """Extract text from HTML files and return text with basic metadata"""
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try to extract title
    title_tag = soup.find('title')
    title = title_tag.get_text() if title_tag else "Untitled HTML Document"
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    text = soup.get_text(separator='\n', strip=True)
    
    # Basic metadata for HTML files
    page_metadata = [{
        "page_number": 1,
        "char_start": 0,
        "char_end": len(text),
        "title": title,
        "file_type": "HTML"
    }]
    
    return text, page_metadata

def extract_text_from_url(url: str) -> tuple[str, list[dict]]:
    """Extract text from a web URL and return text with metadata"""
    # More comprehensive headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    
    try:
        # Create a session for better connection handling
        session = requests.Session()
        session.headers.update(headers)
        
        print(f"Attempting to fetch URL: {url}")
        
        # Make the request with longer timeout and SSL verification options
        response = session.get(
            url, 
            timeout=60,  # Increased timeout
            allow_redirects=True,
            verify=True  # SSL verification
        )
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            text, metadata = _parse_html_content(response.content, url)
            warning_text = f"Warning: Content type '{content_type}' may not be HTML. Attempting to parse anyway.\n\n" + text
            return warning_text, metadata
        
        return _parse_html_content(response.content, url)
        
    except requests.exceptions.SSLError as e:
        print(f"SSL Error for {url}: {str(e)}")
        # Retry without SSL verification
        try:
            print(f"Retrying {url} without SSL verification...")
            session = requests.Session()
            session.headers.update(headers)
            response = session.get(url, timeout=60, allow_redirects=True, verify=False)
            response.raise_for_status()
            return _parse_html_content(response.content, url)
        except Exception as retry_e:
            error_text = f"Error fetching content from {url} (SSL retry failed): {str(retry_e)}"
            return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True, "url": url}]
            
    except requests.exceptions.Timeout as e:
        error_text = f"Timeout error fetching content from {url}: {str(e)}"
        return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True, "url": url}]
        
    except requests.exceptions.ConnectionError as e:
        error_text = f"Connection error fetching content from {url}: {str(e)}"
        return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True, "url": url}]
        
    except requests.exceptions.RequestException as e:
        error_text = f"Request error fetching content from {url}: {str(e)}"
        return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True, "url": url}]
        
    except Exception as e:
        error_text = f"Unexpected error fetching content from {url}: {str(e)}"
        return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True, "url": url}]

def _parse_html_content(content: bytes, url: str) -> tuple[str, list[dict]]:
    """Parse HTML content and extract text with metadata"""
    try:
        # Try multiple parsers for better compatibility
        parsers = ['html.parser', 'lxml', 'html5lib']
        soup = None
        
        for parser in parsers:
            try:
                soup = BeautifulSoup(content, parser)
                break
            except Exception as e:
                print(f"Parser '{parser}' failed: {str(e)}")
                if parser == parsers[-1]:  # Last parser failed
                    raise e
                continue
        
        if soup is None:
            error_text = f"Error: Could not parse HTML content from {url}"
            return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True, "url": url}]
        
        # Try to extract title
        title_tag = soup.find('title')
        title = title_tag.get_text() if title_tag else url
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content areas first
        main_content = None
        content_selectors = [
            'main', 'article', '.content', '#content', '.main', '#main',
            '.post', '.entry', '.article-content', '.page-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                print(f"Found main content using selector: {selector}")
                break
        
        # If no main content found, use the entire body or html
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text content
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up the text
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Skip very short lines
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Limit the text size to prevent extremely large documents
        max_chars = 50000  # 50KB limit
        was_truncated = False
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars] + "\n\n[Content truncated due to size limit]"
            was_truncated = True
        
        final_text = f"Content from {url}:\n\n{cleaned_text}"
        
        # Create metadata
        page_metadata = [{
            "page_number": 1,
            "char_start": 0,
            "char_end": len(final_text),
            "title": title,
            "url": url,
            "file_type": "Web Page",
            "was_truncated": was_truncated
        }]
        
        return final_text, page_metadata
        
    except Exception as e:
        error_text = f"Error parsing HTML content from {url}: {str(e)}"
        return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True, "url": url}]

def detect_file_type(file_path: str) -> str:
    """Detect file type based on extension or content"""
    if file_path.startswith(('http://', 'https://')):
        return 'url'
    
    file_extension = Path(file_path).suffix.lower()
    extension_map = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.txt': 'txt',
        '.html': 'html',
        '.htm': 'html'
    }
    return extension_map.get(file_extension, 'unknown')

def extract_text_from_file(file_path: str) -> tuple[str, list[dict]]:
    """Extract text from various file types and return text with metadata"""
    file_type = detect_file_type(file_path)
    
    extractors = {
        'pdf': extract_text_from_pdf,
        'docx': extract_text_from_docx,
        'doc': extract_text_from_doc,
        'txt': extract_text_from_txt,
        'html': extract_text_from_html_file,
        'url': extract_text_from_url
    }
    
    if file_type in extractors:
        try:
            return extractors[file_type](file_path)
        except Exception as e:
            error_text = f"Error processing {file_type} file: {str(e)}"
            return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True}]
    else:
        error_text = f"Unsupported file type: {file_type}"
        return error_text, [{"page_number": 1, "char_start": 0, "char_end": len(error_text), "error": True}]

def process_document(file_path: str, embedding_model: str = "text-embedding-3-small") -> int:
    """Process document and add to vector store"""
    import sys
    try:
        print(f"Processing document: {file_path} with embedding model: {embedding_model}", flush=True)
        sys.stdout.flush()
        
        text, document_metadata = extract_text_from_file(file_path)
        
        if text.startswith("Error") or not text.strip():
            print(f"Error or empty text for file: {file_path}", flush=True)
            sys.stdout.flush()
            return 0
        
        print(f"Extracted {len(text)} characters, splitting into chunks...", flush=True)
        sys.stdout.flush()
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        
        if not chunks:
            print(f"No chunks created for file: {file_path}", flush=True)
            sys.stdout.flush()
            return 0
        
        print(f"Created {len(chunks)} chunks, getting embeddings...", flush=True)
        sys.stdout.flush()
        
        # Store in Chroma with specified embedding model
        embeddings = get_embeddings(embedding_model)
        print(f"Got embeddings, creating vectorstore...", flush=True)
        sys.stdout.flush()
        
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        
        # Add metadata about the source file and chunk positions
        file_name = Path(file_path).name if not file_path.startswith('http') else file_path
        
        # Create metadata for each chunk, including document metadata where available
        metadatas = []
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            base_metadata = {
                "source": file_name,
                "file_type": detect_file_type(file_path),
                "embedding_model": embedding_model,
                "chunk_index": i,
                "file_path": file_path
            }
            
            # Find the best matching document metadata for this chunk position
            chunk_end_pos = current_pos + len(chunk)
            best_metadata = None
            
            for doc_meta in document_metadata:
                doc_start = doc_meta.get("char_start", 0)
                doc_end = doc_meta.get("char_end", len(text))
                
                # Check if chunk overlaps with this document metadata
                if not (chunk_end_pos <= doc_start or current_pos >= doc_end):
                    best_metadata = doc_meta
                    break
            
            # Add document-specific metadata if found
            if best_metadata:
                if "page_number" in best_metadata:
                    base_metadata["page_number"] = best_metadata["page_number"]
                if "paragraph_number" in best_metadata:
                    base_metadata["paragraph_number"] = best_metadata["paragraph_number"]
                if "estimated_page" in best_metadata:
                    base_metadata["estimated_page"] = best_metadata["estimated_page"]
                if "title" in best_metadata:
                    base_metadata["title"] = best_metadata["title"]
                if "url" in best_metadata:
                    base_metadata["url"] = best_metadata["url"]
            
            metadatas.append(base_metadata)
            current_pos = chunk_end_pos
        
        print(f"Adding {len(chunks)} chunks to vectorstore...", flush=True)
        sys.stdout.flush()
        
        vectorstore.add_texts(chunks, metadatas=metadatas)
        
        print("Persisting vectorstore...", flush=True)
        sys.stdout.flush()
        
        vectorstore.persist()
        
        print(f"Successfully processed document: {file_path}, added {len(chunks)} chunks", flush=True)
        sys.stdout.flush()
        
        return len(chunks)
        
    except Exception as e:
        print(f"Error processing document {file_path}: {str(e)}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        raise e

def clear_vector_database():
    """Clear all data from the vector database and uploaded files"""
    import shutil
    import time
    import gc
    import sqlite3
    
    try:
        vector_db_cleared = False
        uploads_cleared = False
        collections_cleared = False
        
        # Step 1: Try ChromaDB API clearing first
        if os.path.exists(CHROMA_DB_DIR):
            try:
                # Try with default embedding model
                embeddings = get_embeddings("text-embedding-3-small")
                vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
                
                # Get the client and delete all collections
                client = vectorstore._client
                collections = client.list_collections()
                
                collections_deleted = 0
                for collection in collections:
                    print(f"Deleting collection: {collection.name}")
                    try:
                        client.delete_collection(collection.name)
                        collections_deleted += 1
                    except Exception as e:
                        print(f"Error deleting collection {collection.name}: {e}")
                
                # Close and cleanup
                del vectorstore
                del client
                gc.collect()
                time.sleep(1)
                
                if collections_deleted > 0:
                    collections_cleared = True
                    print(f"Successfully cleared {collections_deleted} ChromaDB collections")
                else:
                    print("No collections to clear")
                    collections_cleared = True  # No collections means already clear
                
            except Exception as e:
                print(f"Could not clear via ChromaDB API: {str(e)}")
                # Try direct file approach if API fails
                try:
                    # Delete the main database file
                    db_file = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
                    if os.path.exists(db_file):
                        os.remove(db_file)
                        print("Deleted chroma.sqlite3 directly")
                        collections_cleared = True
                except Exception as e2:
                    print(f"Could not delete database file directly: {e2}")
            
            # Step 2: Complete database reset - remove entire directory (optional for full cleanup)
            directory_removed = False
            try:
                # Force removal of entire directory
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        if os.path.exists(CHROMA_DB_DIR):
                            # Make all files writable
                            for root, dirs, files in os.walk(CHROMA_DB_DIR):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    try:
                                        os.chmod(file_path, 0o777)
                                    except:
                                        pass
                            
                            # Remove the entire directory
                            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
                            time.sleep(0.5)
                            
                            if not os.path.exists(CHROMA_DB_DIR):
                                print(f"Successfully removed vector database directory: {CHROMA_DB_DIR}")
                                directory_removed = True
                                break
                            else:
                                print(f"Attempt {attempt + 1}: Directory still exists, retrying...")
                                time.sleep(1)
                        else:
                            directory_removed = True
                            break
                            
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed: {str(e)}")
                        if attempt < max_attempts - 1:
                            time.sleep(1)
                        else:
                            print("All removal attempts failed - but collections may still be cleared")
                
            except Exception as e:
                print(f"Directory removal failed: {str(e)}")
            
            # Consider it cleared if collections were cleared, even if directory removal failed
            vector_db_cleared = collections_cleared or directory_removed
        else:
            print("Vector database directory doesn't exist - already clear")
            vector_db_cleared = True
        
        # Step 3: Recreate clean directory
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        print(f"Recreated clean vector database directory: {CHROMA_DB_DIR}")
        
        # Step 4: Clear uploaded files
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            files_removed = []
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        files_removed.append(filename)
                        print(f"Removed uploaded file: {filename}")
                    except Exception as e:
                        print(f"Could not remove {filename}: {str(e)}")
            
            if files_removed:
                uploads_cleared = True
        
        return {
            "success": True,
            "vector_db_cleared": vector_db_cleared,
            "uploads_cleared": uploads_cleared,
            "collections_cleared": collections_cleared if 'collections_cleared' in locals() else False,
            "method_used": "collections_api" if collections_cleared else "directory_removal" if vector_db_cleared else "none"
        }
    except Exception as e:
        print(f"Error clearing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

def simple_clear_vector_database():
    """Simple alternative database clearing method that focuses on data removal"""
    try:
        print("Starting simple database clear...")
        
        # Method 1: Clear via ChromaDB API
        if os.path.exists(CHROMA_DB_DIR):
            try:
                embeddings = get_embeddings("text-embedding-3-small")
                vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
                
                # Get collection and clear all documents
                collection = vectorstore._collection
                if collection:
                    # Get all IDs and delete them
                    all_docs = collection.get()
                    if all_docs and 'ids' in all_docs and all_docs['ids']:
                        collection.delete(ids=all_docs['ids'])
                        print(f"Deleted {len(all_docs['ids'])} documents from collection")
                    else:
                        print("Collection is already empty")
                
                del vectorstore
                print("Simple clear completed successfully")
                return {"success": True, "method": "simple_clear", "vector_db_cleared": True}
                
            except Exception as e:
                print(f"Simple clear failed: {e}")
                return {"success": False, "error": str(e)}
        else:
            print("Database directory doesn't exist")
            return {"success": True, "method": "no_database", "vector_db_cleared": True}
            
    except Exception as e:
        print(f"Error in simple clear: {e}")
        return {"success": False, "error": str(e)}

def inspect_database_tables():
    """Inspect the SQLite database tables and their contents"""
    import sqlite3
    
    try:
        db_path = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.exists(db_path):
            return {"error": "Database file does not exist"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        table_info = {}
        for table in tables:
            table_name = table[0]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            schema = cursor.fetchall()
            
            table_info[table_name] = {
                "row_count": count,
                "columns": [col[1] for col in schema]  # col[1] is column name
            }
        
        conn.close()
        return {"tables": table_info}
        
    except Exception as e:
        return {"error": str(e)}

def get_database_status():
    """Get current status of the vector database and uploaded files"""
    try:
        status = {
            "vector_db_exists": False,
            "vector_db_collections": 0,
            "vector_db_documents": 0,
            "collection_names": [],
            "uploaded_files": 0,
            "uploaded_file_list": [],
            "database_file_exists": False,
            "database_completely_clear": False
        }
        
        # Check if database file exists
        db_path = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        status["database_file_exists"] = os.path.exists(db_path)
        
        # If no database file exists, it's completely clear
        if not status["database_file_exists"]:
            status["database_completely_clear"] = True
            status["vector_db_exists"] = False
        else:
            # Check vector database using ChromaDB
            try:
                # Try with default embedding model
                embeddings = get_embeddings("text-embedding-3-small")
                vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
                
                # Get client and check collections
                client = vectorstore._client
                collections = client.list_collections()
                
                status["vector_db_exists"] = True
                status["vector_db_collections"] = len(collections)
                status["collection_names"] = [col.name for col in collections]
                
                # Count total documents across all collections
                total_docs = 0
                for collection in collections:
                    try:
                        count = collection.count()
                        total_docs += count
                        print(f"Collection '{collection.name}' has {count} documents")
                    except Exception as e:
                        print(f"Could not count documents in collection '{collection.name}': {str(e)}")
                
                status["vector_db_documents"] = total_docs
                
                # Consider it completely clear if no collections or no documents
                status["database_completely_clear"] = (len(collections) == 0 or total_docs == 0)
                
                # Cleanup
                del vectorstore
                del client
                
            except Exception as e:
                print(f"Error checking vector database: {str(e)}")
                status["vector_db_exists"] = True  # File exists but can't access properly
                status["database_completely_clear"] = False
        
        # Check uploaded files
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
            status["uploaded_files"] = len(files)
            status["uploaded_file_list"] = files
        
        return status
    except Exception as e:
        return {"error": str(e)}
