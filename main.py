import os
import uuid
import io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
import pypdf # The new PDF reader we just installed

# 1. Load environment variables BEFORE doing anything else
load_dotenv()

from services.embeddings import get_embedding
from services.pinecone_client import get_pinecone_index
from groq import Groq

app = FastAPI()

# 2. Initialize the Groq client 
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class IngestRequest(BaseModel):
    text: str
    source: str = "manual"

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/ingest")
def ingest(request: IngestRequest):
    embedding = get_embedding(request.text)
    index = get_pinecone_index()
    doc_id = str(uuid.uuid4())
    
    index.upsert(vectors=[{
        "id": doc_id,
        "values": embedding,
        "metadata": {
            "text": request.text,
            "source": request.source
        }
    }])
    
    return {"message": "Text ingested successfully", "id": doc_id}

def check_grounding(context: str, answer: str) -> bool:
    """
    Evaluates if the generated answer is strictly based on the provided context.
    """
    evaluator_prompt = f"""You are a strict safety and ethics evaluator. 
    Read the provided context and the generated answer. 
    If the answer contains claims, facts, or numbers not present in the context, output exactly "NO".
    If the answer is completely supported by the context, output exactly "YES".
    
    Context: {context}
    
    Answer: {answer}
    """
    
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": evaluator_prompt}],
        model="llama-3.3-70b-versatile", 
        temperature=0.0 
    )
    
    evaluation = response.choices[0].message.content.strip().upper()
    return "YES" in evaluation

@app.post("/query")
def query(request: QueryRequest):
    # Step A: Retrieval Phase
    embedding = get_embedding(request.question)
    index = get_pinecone_index()
    
    results = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True
    )
    
    contexts = [match["metadata"]["text"] for match in results.get("matches", [])]
    combined_context = "\n\n".join(contexts)
    
    # Step B: Generation Phase
    system_prompt = f"""You are a helpful knowledge base assistant. 
    Use the following retrieved context to answer the user's question accurately. 
    If the answer cannot be found in the context, say "I don't have enough information to answer that."
    
    Context:
    {combined_context}"""
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ],
        model="llama-3.3-70b-versatile", 
    )
    
    answer = chat_completion.choices[0].message.content
    
    # Step C: Safety Check Guardrail
    is_grounded = check_grounding(combined_context, answer)
    
    if not is_grounded:
        answer = "I'm sorry, but I couldn't find a fully verified answer in my knowledge base."
        
    return {
        "question": request.question,
        "answer": answer,
        "is_safe": is_grounded,
        "contexts": contexts
    }

# --- NEW ENDPOINT: PDF UPLOAD ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # 1. Read the uploaded PDF file
        contents = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
        
        # 2. Extract all text from the pages
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
                
        # 3. Chunk the text (Split into paragraphs so Pinecone can digest it)
        # We split by double newlines (paragraphs) and ignore empty ones
        raw_chunks = extracted_text.split("\n\n")
        chunks = [chunk.strip() for chunk in raw_chunks if len(chunk.strip()) > 50]
        
        # 4. Embed and Upload each chunk to Pinecone
        index = get_pinecone_index()
        doc_ids = []
        
        for chunk in chunks:
            embedding = get_embedding(chunk)
            chunk_id = str(uuid.uuid4())
            
            index.upsert(vectors=[{
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": file.filename
                }
            }])
            doc_ids.append(chunk_id)
            
        return {
            "message": f"Successfully processed {file.filename}", 
            "total_chunks_saved": len(doc_ids)
        }
        
    except Exception as e:
        return {"error": str(e)}