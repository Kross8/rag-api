# Knowledge Base RAG API with Safety Guardrails

An advanced REST API backend implementing Retrieval-Augmented Generation (RAG) with a strict focus on AI safety, hallucination prevention, and ethical standards. 

This system allows users to upload raw text or PDF documents, converts them into vector embeddings, and uses a Large Language Model (LLM) to answer questions based strictly on the provided context. It features a built-in "Grounding Evaluator" that actively blocks hallucinated responses, making it highly suitable for critical integrations where accuracy is paramount, such as physical robotics.

## Key Features

* Document Ingestion: Upload full PDF reports or manual text entries. The system automatically extracts, chunks, and processes the text.
* Semantic Search: Utilizes high-speed local embeddings to accurately match user queries with the most relevant document chunks.
* Built-in Safety Guardrail: Implements an "LLM-as-a-Judge" evaluator that intercepts the AI's generated response and verifies it against the retrieved context. If the AI hallucinates or pulls unverified outside information, the API physically blocks the response.
* Blazing Fast: Built on FastAPI, leveraging FastEmbed for zero-API-key local embeddings and Groq's Llama-3 models for near-instant inference.

## Tech Stack

* Framework: FastAPI (Python)
* LLM Engine: Groq (llama-3.3-70b-versatile)
* Vector Database: Pinecone
* Embeddings: FastEmbed (BAAI/bge-small-en-v1.5)
* Document Processing: PyPDF

## Quick Start Setup

1. Clone the repository
```bash
git clone [https://github.com/Kross8/rag-api.git](https://github.com/Kross8/rag-api.git)
cd rag-api
```
2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```
3. Configure Environment Variables
## Create a .env file in the root directory and add your API keys:
### GROQ_API_KEY=your_groq_api_key
### PINECONE_API_KEY=your_pinecone_api_key
### PINECONE_INDEX_NAME=your_index_name
## Run the Server
```bash
python -m uvicorn main:app --reload
```
## Access the interactive Swagger UI API documentation at: https://www.google.com/search?q=http://127.0.0.1:8000/docs
## API Endpoints
### POST /upload
Upload a standard PDF document. The API will extract the text, split it into paragraphs, generate vectors, and upload them to Pinecone.

### POST /ingest
Manually add specific text facts to the vector database.
{
  "text": "When integrating Large Language Models into physical robotics, strict safety standards dictate that a secondary evaluator model must verify all execution commands.",
  "source": "Robotics Safety Standards Report"
}
### POST /query
Ask a question. The API will retrieve the context, generate an answer, run the safety evaluator, and return the result.
{"question": "What is required when integrating LLMs into physical robots?"}
## Example Response:
{"question": "What is required when integrating LLMs into physical robots?",
  "answer": "According to the provided context, strict safety standards dictate that a secondary evaluator model must verify all physical execution commands.",
  "is_safe": true,
  "contexts": [
    "When integrating Large Language Models into physical robotics..."
  ]
}
