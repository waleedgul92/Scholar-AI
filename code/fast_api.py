# Corrected Line 1: Import from 'fastapi' library
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # To allow requests from your UI
from pydantic import BaseModel, HttpUrl , Field # For request/response validation
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document 
import logging
import uvicorn
import shutil
import os
import tempfile 
from llm_model import load_gemini_model # Assuming these are in the same package
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from document_handler import load_pdf_document, split_documents , embed_data

from chatbot import get_response
from document_handler import load_pdf_document, split_documents, embed_data, retrieve_contextual_data 
# from langchain_community.document_loaders import YoutubeLoader
from research import get_research_papers , get_research_papers_from_arxiv
import time
import os

import chromadb
import gc



def rmtree_with_retry(path: str, max_retries: int = 5, delay_s: float = 1.5): # Increased retries and delay
    """
    Attempts to remove a directory tree, with retries on PermissionError.
    """
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            logger.info(f"Successfully deleted directory: {path}")
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} to delete {path} failed with PermissionError: {e}. Retrying in {delay_s}s...")
                time.sleep(delay_s)
            else:
                logger.error(f"Failed to delete {path} after {max_retries} attempts due to PermissionError: {e}")
                raise
        except FileNotFoundError:
            logger.info(f"Directory not found, presumed already deleted: {path}")
            return
        except Exception as e:
            logger.error(f"Failed to delete {path} due to an unexpected error: {e}", exc_info=True)
            raise


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ResLLM Backend Service",
    description="API for YouTube Transcripts and LLM Chat.",
)
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:63342", # Common port for PyCharm/WebStorm local server
    "null", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
gemini_model = load_gemini_model()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    mode: str = "youtube"
    history: List[ChatMessage] = []
    model_name: str = "gemini-pro" # Matches existing UI call

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
class ChatRequest(BaseModel):
    query: str = Field(..., description="User's query")
    context: Optional[str] = Field(None, description="Context for the query (e.g., transcript snippet, document text)")
    mode: str = Field("youtube", description="Mode of operation (default: youtube, research_paper)")
    history: List[ChatMessage] = Field([], description="Chat history") # Add history field
    model_name: str = Field("gemini", description="Name of the LLM model to use (default: gemini)") # Add model_name field

class chat_response(BaseModel):
    response: str = Field(..., description="Response from the LLM")

class ResearchPaperRequest(BaseModel):
    query: str = Field(..., description="User query for research papers")

class ResearchPaperResponse(BaseModel):
    titles: List[str] = Field(..., description="List of research paper titles")

class PaperRequest(BaseModel):
    papers: List[str]


class ResetRequest(BaseModel):
    context_type: str


class RetrieveDataRequest(BaseModel):
    query: str
    mode: str

@app.get("/")
async def root():
    return {"message": "Welcome to the ResLLM Backend Service!"}

    

@app.post("/get_papers")
async def get_papers(request: PaperRequest) -> dict:
    """
    Get research papers based on the user's query.
    """
    try:
        logger.info(f"Received request to process and embed papers: {request.papers}")
        if not request.papers:
            raise HTTPException(status_code=400, detail="No paper titles provided.")

        # 1. Fetch full content for these titles
        paper_contents_str_list = get_research_papers_from_arxiv(request.papers)

        # 2. Convert these strings to Document objects, adding titles as metadata
        paper_documents: List[Document] = []
        for i, content_str in enumerate(paper_contents_str_list):
            title = request.papers[i] if i < len(request.papers) else "Unknown Title"
            paper_documents.append(Document(page_content=content_str, metadata={"source": title}))
        
        if not paper_documents:
            logger.warning("No content retrieved for any of the specified ArXiv papers.")
            # Return a message indicating no content was processed
            return {"message": "No content found for the specified ArXiv papers. Nothing embedded.", "context_type": "research_paper", "papers": request.papers}

        # 3. Split these documents
        split_paper_docs = split_documents(paper_documents, mode="research_paper")
        
        if not split_paper_docs:
            logger.warning(f"Research papers resulted in no splittable documents after processing: {request.papers}")
             # If no docs to embed, it's not an embedding failure.
            return {"message": "Research papers processed, but no content was suitable for embedding.", "context_type": "research_paper", "papers": request.papers}


        # 4. Embed these split documents
        success_embedding = embed_data(split_paper_docs, model_name="gemini", mode="research_paper")

        if success_embedding:
            return {"message": "Selected research papers processed and embedded successfully.", "context_type": "research_paper", "papers": request.papers}
        else:
            raise HTTPException(status_code=500, detail="Failed to embed research paper data into vector store.")

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error in /get_papers endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



@app.post("/research", response_model=ResearchPaperResponse)
async def fetch_research_papers(request: ResearchPaperRequest):
    try:
        result = get_research_papers(request.query)
        return ResearchPaperResponse(titles=result.titles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_pdf")
async def process_uploaded_pdf(file: UploadFile = File(...)) -> dict:
    temp_file_path = None
    try:
        # Ensure /tmp directory exists or use a more robust temp file solution
        temp_dir = "/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        temp_file_path = os.path.join(temp_dir, file.filename if file.filename else "uploaded_temp.pdf")
        content = await file.read() # Read async
        with open(temp_file_path, 'wb') as f: # Write sync
            f.write(content)
        
        logger.info(f"PDF '{file.filename}' saved temporarily to {temp_file_path}")
        
        raw_documents = load_pdf_document(temp_file_path)
        if not raw_documents:
            raise HTTPException(status_code=400, detail="PDF content could not be loaded or is empty.")

        split_docs = split_documents(raw_documents, mode="pdf")
        if not split_docs:
            # This case might mean the PDF was very small or content was filtered out
            logger.warning(f"PDF '{file.filename}' resulted in no splittable documents after processing.")
            # Depending on desired behavior, you might still return success or a specific message.
            # For now, let's assume if no split_docs, embedding is not attempted.

        if split_docs: # Only embed if there are documents to embed
            success_embedding = embed_data(split_docs, model_name="gemini", mode="pdf")
            if not success_embedding:
                raise HTTPException(status_code=500, detail="Failed to embed PDF data into vector store.")
        else:
            logger.info(f"No documents to embed for PDF '{file.filename}'. Skipping embedding step.")
            # If no docs to embed, it's not an embedding failure, but processing might be considered incomplete by some standards.
            # Let's treat it as partial success for now.
        
        return {"message": f"PDF '{file.filename}' processed. Embeddings updated if content was found.", "filename": file.filename, "context_type": "pdf"}

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded PDF '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e_remove:
                logger.error(f"Error cleaning up temporary file {temp_file_path}: {e_remove}")

@app.post("/chat")
async def chat_endpoint(request_data: ChatRequest) -> Dict[str, str]:
    """
    Handles chat requests, using the new get_response_from_llm logic.
    """
    try:

        logger.info(f"Chat request received. Mode: {request_data.mode}, Query: '{request_data.query}', History length: {len(request_data.history)}")
        if request_data.context:
            logger.info(f"Context provided (first 100 chars): {request_data.context[:100]}...")
        
        # Call the user's provided get_response function
        response_text = get_response(
            model=gemini_model,  # Use the initialized model
            query=request_data.query,
            context=request_data.context,
            mode=request_data.mode,
            history=request_data.history
        )
        
        return {"response": response_text}

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred in chat processing: {str(e)}")
    

@app.post("/reset_embeddings")
async def reset_embeddings_endpoint(request: ResetRequest) -> dict:
    try:
        context_type_to_reset = request.context_type.lower()
        reset_messages = []
        not_found_messages = []
        logger.info(f"Received request to clear embeddings data for: '{context_type_to_reset}'")

        paths_to_process = []
        if context_type_to_reset == "pdf" or context_type_to_reset == "all":
            paths_to_process.append(("./chroma_langchain_db_pdf", "PDF Embeddings"))
        if context_type_to_reset == "research_paper" or context_type_to_reset == "all":
            paths_to_process.append(("./chroma_langchain_db_research_paper", "Research Paper Embeddings"))

        if not paths_to_process and context_type_to_reset not in ["all", "pdf", "research_paper"]:
            raise HTTPException(status_code=400, detail=f"Invalid context_type '{request.context_type}'.")

        any_action_taken = False
        for db_path, type_name in paths_to_process:
            try:
                if os.path.exists(db_path) and os.path.isdir(db_path):
                    logger.info(f"Deleting directory for {type_name}: {db_path}")
                    rmtree_with_retry(db_path)
                    reset_messages.append(f"{type_name} directory deleted.")
                    any_action_taken = True
                else:
                    logger.info(f"{type_name} directory not found at {db_path}")
                    not_found_messages.append(f"{type_name} store not found.")
            except Exception as e:
                logger.error(f"Failed to delete directory for {type_name} at {db_path}: {e}", exc_info=True)
                if context_type_to_reset != "all":
                    raise HTTPException(status_code=500, detail=f"Failed to reset {type_name} store: {str(e)}")

        final_message_parts = []
        if reset_messages:
            final_message_parts.extend(reset_messages)
        if not_found_messages and not reset_messages:
            final_message_parts.extend(not_found_messages)
        if not any_action_taken and not final_message_parts:
            final_message_parts.append("No active embeddings found to reset for the specified type(s).")

        return {"message": " ".join(final_message_parts).strip() or "Reset action processed."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /reset_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    

@app.post("/retrieve_contextual_data", response_model=List[Dict[str, Any]])
async def handle_retrieve_data(request: RetrieveDataRequest):
    """
    Endpoint to retrieve contextual data using MultiQueryRetriever.
    """
    try:
        
        # Call the user's (now modified) retrieval logic
        results = retrieve_contextual_data(
            query=request.query,
            mode=request.mode,
            llm_instance=gemini_model # Pass the initialized LLM
        )
        if not results:
            logger.info(f"No results retrieved for query '{request.query}' in mode '{request.mode}'.")
            # Return empty list, which is fine. Frontend can handle "no results".
        return results
    except ValueError as ve: # Catch specific errors like missing API key more gracefully
        logger.error(f"ValueError in /retrieve_contextual_data: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /retrieve_contextual_data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during data retrieval: {str(e)}")


if __name__ == "__main__":
    
    logger.info("Starting Uvicorn server directly from script on http://127.0.0.1:8000")
    # Note: Running this way might have issues with --reload compared to command line
    uvicorn.run(app, host="127.0.0.1", port=8000)
