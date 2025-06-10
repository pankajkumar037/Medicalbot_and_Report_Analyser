# Medicalbot_and_Report_Analyser

This project provides a RAG (Retrieval Augmented Generation) Chatbot API and a Medical Report Analyser. The chatbot can answer medical queries by retrieving information from a vector store, and the report analyser can process PDF medical reports to extract and analyze information.

## Features

*   **RAG Chatbot:** Answers medical questions using a pre-built knowledge base.
*   **Medical Report Analyser:** Upload PDF medical reports for analysis.
*   **FastAPI:** Robust and high-performance API backend.

## Setup

### Prerequisites

*   Python 3.9+
*   `pip` (Python package installer)

### Environment Variables

Create a `.env` file in the root directory of the project and add the following environment variables:

```
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
```

*   **`GOOGLE_API_KEY`**: Obtain this from the Google Cloud Console for Google Generative AI Embeddings.
*   **`PINECONE_API_KEY`**: Obtain this from your Pinecone account for vector store operations.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Medicalbot_and_Report_Analyser.git
    cd Medicalbot_and_Report_Analyser
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the FastAPI application, run the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be accessible at `http://0.0.0.0:8000`.

## API Endpoints

### 1. Chat Endpoint

*   **URL**: `/chat`
*   **Method**: `POST`
*   **Description**: Interacts with the RAG chatbot to get answers to medical queries.
*   **Request Body**:
    ```json
    {
        "question": "string",
        "history": [
            {
                "question": "string",
                "answer": "string"
            }
        ]
    }
    ```
    *   `question` (required): The medical question you want to ask.
    *   `history` (optional): A list of previous conversation turns to maintain context.
*   **Response**:
    ```json
    {
        "answer": "string"
    }
    ```
    *   `answer`: The chatbot's response to the question.

### 2. Upload PDF for Analysis

*   **URL**: `/upload-pdf`
*   **Method**: `POST`
*   **Description**: Uploads a PDF medical report for analysis.
*   **Request Body**: `multipart/form-data`
    *   `file` (required): The PDF file to upload.
*   **Response**:
    ```json
    {
        "filename": "string",
        "content_type": "string",
        "analysis_result": "string"
    }
    ```
    *   `filename`: The name of the uploaded file.
    *   `content_type`: The MIME type of the uploaded file (should be `application/pdf`).
    *   `analysis_result`: The analysis generated from the medical report.

### 3. Root Endpoint

*   **URL**: `/`
*   **Method**: `GET`
*   **Description**: Welcome message for the API.
*   **Response**:
    ```json
    {
        "message": "Welcome to the RAG Chatbot API. Use the /chat endpoint to interact with the chatbot."
    }
    ```

### 4. Health Check

*   **URL**: `/health`
*   **Method**: `GET`
*   **Description**: Checks the health status of the API.
*   **Response**:
    ```json
    {
        "status": "healthy"
    }
    ```