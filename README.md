# Scholar-AI


## Overview

Scholar-AI is a research assistant application designed to streamline the process of working with research papers and PDF documents. It leverages Large Language Models (LLMs) to provide a chat-based interface for users to ask questions, retrieve information, and interact with their documents. The application supports fetching research papers from ArXiv, processing uploaded PDF files, and embedding document content for efficient contextual retrieval.

## Features

* **Chat Interface**: Interact with an AI assistant to get answers based on provided documents or general knowledge.
* **Research Paper Retrieval**: Search for research papers on ArXiv by topic.
* **Paper Processing & Embedding**: Process selected research papers by fetching their content, splitting them into manageable chunks, and embedding them for semantic search.
* **PDF Document Handling**:
    * Upload PDF documents.
    * Process and split PDF content.
    * Embed PDF content for contextual querying.
* **Contextual Q&A**: Ask questions about your uploaded PDFs or selected research papers, with the AI providing answers based on the retrieved context from the documents.
* **Multiple Modes**:
    * **Research Paper Mode**: For finding and interacting with ArXiv papers.
    * **PDF Mode**: For interacting with uploaded PDF documents.
* **LLM Integration**: Utilizes Google's Gemini models for chat responses and other language understanding tasks.
* **Vector Store**: Uses ChromaDB for storing and retrieving document embeddings.
* **Dark/Light Theme**: User interface supports toggling between dark and light themes.
* **Reset Functionality**: Clear chat history and reset document embeddings.
* **Speech-to-Text**: Input queries via voice using the browser's Speech Recognition API.

## Technologies Used

* **Backend**:
    * Python
    * FastAPI (for creating the RESTful API)
    * Langchain (for LLM integration, document processing, and retrieval)
    * ChromaDB (vector store for embeddings)
    * Google Generative AI (for Gemini LLM and embeddings)
    * Arxiv API (for fetching research papers)
    * PyMuPDF (for loading PDF documents)
    * Uvicorn (ASGI server)
* **Frontend**:
    * HTML
    * CSS
    * JavaScript
* **Key Python Libraries**: (A more extensive list can be found in `requirements.txt`)
    * `fastapi`
    * `langchain`, `langchain-core`, `langchain-google-genai`, `langchain-community`
    * `chromadb`
    * `google-generativeai`
    * `arxiv`
    * `pypdf` / `pymupdf` (PyMuPDF is used in `document_handler.py`)
    * `uvicorn`
    * `python-dotenv`

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd scholar-ai
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up API Keys**:
    * Create a `keys.env` file in the root directory (or where `llm_model.py` and `document_handler.py` expect it, which appears to be the `code` directory or root based on `load_dotenv("keys.env")`).
    * Add your Google API key to `keys.env`:
        ```env
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        ```
5.  **Run the backend server**:
    The FastAPI application is defined in `code/fast_api.py`.
    ```bash
    python code/fast_api.py
    ```
    This will typically start the server on `http://127.0.0.1:8000`.

6.  **Access the frontend**:
    Open the `main.html` file in your web browser.

## Usage

1.  **Open `main.html` in your browser.**
2.  **Select Mode**:
    * **Research Paper Mode**:
        * Enter a topic in the chat input.
        * The system will suggest relevant paper titles from ArXiv.
        * Select the papers you are interested in and click "Process Selected".
        * Once processed, you can ask questions about the content of these papers.
    * **PDF Mode**:
        * Click the attachment icon (📎) to upload a PDF file.
        * Once the PDF is processed and embedded, you can ask questions about its content.
3.  **Chat**:
    * Type your query in the input field or use the microphone button for voice input.
    * Press Enter or click the send button (▶) to submit your query.
    * The AI assistant will respond based on the selected mode and available context (from processed papers or PDF).
4.  **Theme**: Toggle between light and dark mode using the 🌙/☀️ button.
5.  **Reset**: Click the 🧹 button to clear the chat, and reset embeddings for the selected context type (or all).

## File Structure (Key Files)

* `README.md`: This file.
* `main.html`: The main HTML file for the user interface.
* `requirements.txt`: A list of Python dependencies for the project.
* `code/`: Directory containing the Python backend logic.
    * `fast_api.py`: Defines the FastAPI application, API endpoints, and orchestrates backend logic.
    * `chatbot.py`: Contains the logic for generating responses from the LLM, including prompt construction.
    * `document_handler.py`: Handles PDF loading, document splitting, embedding, and context retrieval from the vector store (ChromaDB).
    * `llm_model.py`: Loads the LLM (e.g., Gemini).
    * `research.py`: Handles fetching research paper suggestions and retrieving content from ArXiv.
* `keys.env` (user-created): Stores API keys (e.g., `GOOGLE_API_KEY`).

## Potential Future Enhancements

* Support for more document types beyond PDF.
* Integration with other research paper databases (e.g., PubMed, Semantic Scholar).
* More advanced chat history management.
* User accounts and persistent storage of processed documents per user.
* Enhanced UI/UX for managing and interacting with documents.
* Support for other LLMs.
