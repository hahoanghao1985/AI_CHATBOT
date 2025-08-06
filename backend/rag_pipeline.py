# RAG logic using LangChain
# Placeholder for Retrieval-Augmented Generation pipeline

# TODO: Implement RAG pipeline with LangChain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
# from langchain_community.document_compressors import CohereRerank  # Import error - will handle dynamically
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHROMA_DB_DIR = "vector_db"

def detect_language(text: str) -> str:
    """Detect language of the input text"""
    if not text or len(text.strip()) == 0:
        return "english"
    
    # Simple language detection based on common patterns
    vietnamese_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text.lower())
    
    # Count Vietnamese characters
    vietnamese_count = len(vietnamese_chars)
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars > 0 and vietnamese_count / total_chars > 0.05:  # If more than 5% are Vietnamese chars
        return "vietnamese"
    
    # Check for common Vietnamese words (expanded list)
    vietnamese_words = [
        'là', 'của', 'và', 'trong', 'có', 'được', 'với', 'này', 'cho', 'từ', 'một', 'các', 'người', 'không', 
        'tôi', 'bạn', 'gì', 'như', 'thế', 'nào', 'về', 'khi', 'đã', 'sẽ', 'để', 'những', 'sau', 'theo', 
        'cũng', 'lại', 'hay', 'nhiều', 'việc', 'qua', 'vào', 'ra', 'lên', 'xuống', 'trên', 'dưới', 
        'ngoài', 'trong', 'bên', 'giữa', 'cần', 'phải', 'nên', 'có thể', 'sao', 'đây', 'đó', 'kia',
        'bao', 'mấy', 'đâu', 'ai', 'cái', 'con', 'chiếc', 'làm', 'xem', 'biết', 'hiểu', 'nói', 'viết'
    ]
    
    # Normalize text for word matching
    normalized_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = normalized_text.split()
    
    if len(words) == 0:
        return "english"
    
    vietnamese_word_count = sum(1 for word in words if word in vietnamese_words)
    word_ratio = vietnamese_word_count / len(words)
    
    if word_ratio > 0.1:  # If more than 10% are Vietnamese words
        return "vietnamese"
    
    # Check for Vietnamese question words and common phrases
    vietnamese_patterns = [
        r'\b(làm sao|thế nào|như thế nào|ra sao)\b',
        r'\b(là gì|gì là|cái gì)\b', 
        r'\b(ở đâu|đâu là|tại đâu)\b',
        r'\b(khi nào|lúc nào|bao giờ)\b',
        r'\b(tại sao|vì sao|sao lại)\b',
        r'\b(có phải|phải không|đúng không)\b'
    ]
    
    for pattern in vietnamese_patterns:
        if re.search(pattern, text.lower()):
            return "vietnamese"
    
    return "english"

def get_language_specific_prompt(language: str) -> str:
    """Get prompt template based on detected language"""
    if language == "vietnamese":
        return """Bạn là một trợ lý AI hỗ trợ. Sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi.
Nếu câu trả lời không có trong ngữ cảnh, hãy nói "Tôi không biết".
Luôn trả lời bằng tiếng Việt và tham khảo các nguồn cụ thể khi có thể.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Hướng dẫn:
1. Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp
2. Hãy chính xác và ngắn gọn
3. Nếu ngữ cảnh không chứa đủ thông tin để trả lời đầy đủ câu hỏi, hãy nói rõ
4. Tham khảo các tài liệu hoặc trang cụ thể khi có liên quan
5. Trả lời bằng tiếng Việt

Trả lời:"""
    else:
        return """You are an AI assistant that answers questions based on the provided context. 
Always provide accurate answers based on the context and include source references when possible.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based on the provided context
2. Be accurate and concise
3. If the context doesn't contain enough information to answer the question fully, say so
4. Reference specific documents or pages when relevant
5. Respond in English

Answer:"""

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

def get_vectorstore(embedding_model: str = "text-embedding-3-small"):
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=get_embeddings(embedding_model))

def get_reranked_retriever(vectorstore, k: int = 3, reranker_type: str = "none"):
    """Get a retriever with optional re-ranking"""
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})  # Get more docs for re-ranking
    
    if reranker_type == "cohere":
        try:
            # Dynamic import for CohereRerank from langchain-cohere
            from langchain_cohere import CohereRerank
            # Note: Requires COHERE_API_KEY environment variable
            compressor = CohereRerank(model="rerank-english-v3.0")
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        except ImportError as e:
            print(f"CohereRerank not available (import error): {str(e)}")
            print("Make sure langchain-cohere is installed: pip install langchain-cohere")
            print("And set COHERE_API_KEY environment variable")
            print("Falling back to basic retriever")
            return vectorstore.as_retriever(search_kwargs={"k": k})
        except Exception as e:
            print(f"Cohere reranker not available: {str(e)}")
            print("Falling back to basic retriever")
            return vectorstore.as_retriever(search_kwargs={"k": k})
    
    elif reranker_type == "llm":
        try:
            # Use LLM-based compression/extraction
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            compressor = LLMChainExtractor.from_llm(llm)
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        except Exception as e:
            print(f"LLM compressor not available: {str(e)}")
            print("Falling back to basic retriever")
            return vectorstore.as_retriever(search_kwargs={"k": k})
    
    else:
        # No re-ranking, just basic retrieval
        return vectorstore.as_retriever(search_kwargs={"k": k})

def compress_context_if_needed(context: str, max_tokens: int = 3000) -> str:
    """Compress context if it's too long"""
    # Simple estimation: ~4 characters per token
    estimated_tokens = len(context) / 4
    
    if estimated_tokens <= max_tokens:
        return context
    
    print(f"Context too long ({estimated_tokens:.0f} tokens), compressing...")
    
    # Split into chunks and summarize
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Calculate target length
        target_length = int(max_tokens * 3)  # ~3 chars per token for compressed text
        
        if len(context) > target_length:
            # Take the first portion and compress it
            truncated_context = context[:target_length * 2]  # Get more to summarize
            
            summary_prompt = f"""Please summarize the following text, keeping the most important information:

{truncated_context}

Summary:"""
            
            summary = llm.predict(summary_prompt)
            return summary
        else:
            return context
            
    except Exception as e:
        print(f"Context compression failed: {str(e)}")
        # Fallback: simple truncation
        target_length = int(max_tokens * 3)
        return context[:target_length] + "...[truncated]"

def get_answer(
    query: str, 
    model: str = "gpt-3.5-turbo", 
    embedding_model: str = "text-embedding-3-small",
    chunk_count: int = 3,
    reranker_type: str = "none",
    use_compression: bool = True
) -> dict:
    """Get answer with configurable retrieval parameters and source references"""
    # Create LLM with specified model
    llm = ChatOpenAI(model=model, temperature=0)
    
    # Get vectorstore with specified embedding model
    vectorstore = get_vectorstore(embedding_model)
    
    # Get retriever with re-ranking
    retriever = get_reranked_retriever(vectorstore, k=chunk_count, reranker_type=reranker_type)
    
    # Get documents for source references
    docs = retriever.get_relevant_documents(query)
    
    # Extract source references
    sources = []
    seen_sources = set()
    
    # Detect language of the question
    detected_language = detect_language(query)
    
    # Also check document content language if available
    if docs and len(docs) > 0:
        # Sample some document content to detect language
        sample_content = " ".join([doc.page_content[:200] for doc in docs[:3]])  # Sample from first 3 docs
        doc_language = detect_language(sample_content)
        
        # If document language is Vietnamese and question language detection is uncertain, prefer Vietnamese
        if doc_language == "vietnamese":
            detected_language = "vietnamese"
    
    print(f"Detected language: {detected_language}")
    
    for doc in docs:
        metadata = doc.metadata
        source_info = {
            "file_name": metadata.get("source", "Unknown"),
            "file_path": metadata.get("file_path", ""),
            "page_number": metadata.get("page_number"),
            "paragraph_number": metadata.get("paragraph_number"),
            "estimated_page": metadata.get("estimated_page", False),
            "title": metadata.get("title"),
            "url": metadata.get("url"),
            "file_type": metadata.get("file_type", "unknown")
        }
        
        # Create a unique identifier for this source
        source_key = f"{source_info['file_name']}_{source_info.get('page_number', 'no_page')}"
        
        if source_key not in seen_sources:
            sources.append(source_info)
            seen_sources.add(source_key)
    
    # Get language-specific prompt template
    enhanced_prompt_template = get_language_specific_prompt(detected_language)
    
    if use_compression and reranker_type == "none":
        # Apply compression if not using re-ranking (to avoid double processing)
        try:
            # Combine context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Compress if needed
            compressed_context = compress_context_if_needed(context)
            
            # Create a custom prompt with compressed context
            final_prompt = enhanced_prompt_template.format(context=compressed_context, question=query)
            
            answer = llm.predict(final_prompt)
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(docs),
                "compression_used": True,
                "language_detected": detected_language
            }
            
        except Exception as e:
            print(f"Compression failed, falling back to standard retrieval: {str(e)}")
            # Fallback to standard QA chain
            pass
    
    # Standard QA chain (with or without re-ranking)
    enhanced_prompt = PromptTemplate(
        template=enhanced_prompt_template, 
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": enhanced_prompt}
    )
    
    answer = qa_chain.run(query)
    
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(docs),
        "compression_used": False,
        "language_detected": detected_language
    }
