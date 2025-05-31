import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document # Import Document type hint
from typing import List,Union # Import List type hint
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from langchain_community.vectorstores import FAISS 
import chromadb
from dotenv import load_dotenv
from fastapi import HTTPException
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv("keys.env")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    logger.warning("GOOGLE_API_KEY not found in environment variables.")

def load_pdf_document(file_path: str) -> List[Document]:
    logger.info(f"Loading PDF document from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load() # Returns list of Documents, often one per page
    logger.info(f"Loaded {len(documents)} pages/documents from PDF.")
    # No try/except as requested
    return documents

def split_documents(
    documents: Union[List[Document], List[str]],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    mode: str = "unknown"
) -> List[Document]:
    processed_documents: List[Document] = []
    if not documents:
        logger.warning(f"No documents provided to split for mode: {mode}.")
        return []

    # Check if the first element is a string or Document to determine type
    # This assumes the list is not mixed, which should be the case.
    if isinstance(documents[0], str):
        logger.info(f"Converting list of strings to Document objects for mode: {mode}.")
        # Cast to List[str] for type checker if it was Union
        processed_documents = [Document(page_content=str(doc_content)) for doc_content in documents]
    elif isinstance(documents[0], Document):
        logger.info(f"Using provided list of Document objects for mode: {mode}.")
        # Cast to List[Document] for type checker
        processed_documents = [doc for doc in documents] # Ensure it's a new list if modification is intended later
    else:
        logger.error(f"Unsupported document type for splitting: {type(documents[0])} in mode: {mode}")
        raise ValueError("Documents must be a list of strings or a list of Langchain Document objects.")

    if not processed_documents:
        logger.warning(f"No processable documents after type check for mode: {mode}.")
        return []

    logger.info(f"Splitting {len(processed_documents)} documents into chunks for mode: {mode}. Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    split_docs = text_splitter.split_documents(processed_documents)
    logger.info(f"Split into {len(split_docs)} chunks for mode: {mode}.")
    return split_docs



def embed_data(split_docs: List[Document], model_name="gemini",mode="pdf"):
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    embedding_model = None  
    if model_name == "gemini":
        # Load the Gemini model
        embedding_model= GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


    elif model_name == "huggingface":
        # Load the HuggingFace model
        pass
    persist_directory = f"./chroma_langchain_db_{mode}"
    # logger.info(f"Initializing ChromaDB for mode '{mode}' at {persist_directory} with {len(split_docs)} documents.")

    try:
        vector_store = Chroma(
            collection_name=f"collection_{mode}", # Dynamic collection name
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
        vector_store.add_documents(documents=split_docs)
        vector_store.persist() # Ensure data is saved
        logger.info(f"Successfully added {len(split_docs)} documents to ChromaDB for mode '{mode}'. Data persisted.")
        return True
    except Exception as e:
        logger.error(f"Error during ChromaDB operations for mode '{mode}': {e}", exc_info=True)
        return False

def retrieve_contextual_data(query: str, mode: str, llm_instance: Any) -> List[Dict[str, Any]]:
    """
    Retrieves relevant document chunks from the vector store for a given query and mode.
    Uses MultiQueryRetriever.
    """

    current_google_api_key = os.environ.get("GOOGLE_API_KEY")
    user_provided_api_in_function = google_api_key# From user's function, treat as placeholder

    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key="AIzaSyDUyGcpuxX43s9nUHvv0dOWgAfsLuU3khs")
    except Exception as e:
        logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings in retrieve_contextual_data: {e}")
        # Consider raising HTTPException or returning an empty list with an error message.
        raise HTTPException(status_code=500, detail=f"Embedding model init failed: {str(e)}")

    persist_directory = f"./chroma_langchain_db_{mode}"
    dynamic_collection_name = f"collection_{mode}" # Use dynamic collection name

    if not os.path.exists(persist_directory):
        logger.warning(f"Vector store directory not found for mode '{mode}' at {persist_directory}. No data to retrieve.")
        return [] # Return empty list if DB directory doesn't exist

    try:
        vector_store = Chroma(
            collection_name=dynamic_collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
    except Exception as e:
        logger.error(f"Failed to load Chroma vector store for mode '{mode}' from {persist_directory}: {e}")
        return [] # Return empty if store can't be loaded

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Initialize LLM for MultiQueryRetriever
    # The 'llm_instance' is passed to this function.
    if not llm_instance:
        logger.error("LLM instance not provided to retrieve_contextual_data.")
        raise ValueError("LLM instance is required for MultiQueryRetriever.")

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm_instance
    )
    
    logger.info(f"Invoking MultiQueryRetriever for mode '{mode}' with query: '{query}'")
    try:
        results_mq_docs: List[Document] = multi_query_retriever.invoke(query)
    except Exception as e:
        logger.error(f"Error during MultiQueryRetriever invocation for mode '{mode}': {e}", exc_info=True)
        return []


    # Serialize Document objects to a list of dictionaries
    serialized_results = []
    for doc in results_mq_docs:
        serialized_results.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    logger.info(f"Retrieved {len(serialized_results)} chunks for mode '{mode}'.")
    return serialized_results
