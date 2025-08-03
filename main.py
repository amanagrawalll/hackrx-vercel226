# main.py
import os
import requests
import numpy as np
from fastapi import FastAPI, HTTPException, Header, APIRouter
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from pypdf import PdfReader
from openai import OpenAI

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App and Router Setup ---
app = FastAPI(title="GPT-4 Q&A Service")
router = APIRouter(prefix="/api/v1")

# --- Helper Functions ---
def process_document(url: str):
    """Downloads and chunks the document."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        
        chunks = []
        chunk_size, chunk_overlap = 2000, 300
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - chunk_overlap
        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        print(f"Error processing document: {e}")
        return []

def get_embeddings(texts: List[str], client: OpenAI, model="text-embedding-3-small"):
   """Generates embeddings for a list of texts using OpenAI's API."""
   texts = [text.replace("\n", " ") for text in texts]
   response = client.embeddings.create(input=texts, model=model)
   return [item.embedding for item in response.data]

def generate_answer_with_gpt4(question: str, context: str, client: OpenAI):
    """Generates an answer using the provided OpenAI client and GPT-4."""
   rompt = f"""
    You are an AI assistant specializing in detailed policy and contract analysis. 
    Your task is to provide a clear, brief and factual answer to the `QUESTION` based *only* on the `CONTEXT` provided.

    **Instructions for your response:**

    1.  **Be Subtle:** If the question can be answered in a single line, try to answer it in a single sentence only. Add lines only when necessary information about the points to answer the question aren't included in the first sentence. 

    2.  **Use Complete Sentences:** Always formulate your answer in formal, well-structured sentences. Do not use bullet points unless the source text uses them.

    3.  **Answer Directly:**
       * For questions that can be answered with a "yes" or "no", you must start your response immediately with "Yes," or "No," followed by a very short explanation of 1 or 2 lines.
       * **Crucially, do NOT use any introductory phrases or preambles.** Avoid phrases like "According to the provided document...", "The context states that...", or "Based on the text...".

    4.  **Handle Missing Information:** If the answer to the `QUESTION` absolutely cannot be found in the `CONTEXT`, you must respond with the single phrase: "The information for this question is not available in the provided text."

    CONTEXT:
    ---
    {context}
    ---

    QUESTION:
    {question}

    ANSWER:
    """
   
    try:
        # Using gpt-4o as it's the latest, most powerful, and cost-effective model in the GPT-4 class.
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        if "authentication" in str(e).lower():
            raise HTTPException(status_code=401, detail="Authentication failed with OpenAI. Check your API key.")
        raise HTTPException(status_code=500, detail="Failed to generate answer from OpenAI.")

# --- API Endpoint using the Router ---
@router.post("/hackrx/run", response_model=HackRxResponse, tags=["HackRx"])
async def run_submission(
    request: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    api_key = authorization.split(" ")[1]
    if not api_key:
        raise HTTPException(status_code=401, detail="Bearer token is empty.")

    try:
        # Initialize the OpenAI client for this specific request
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize OpenAI client: {e}")

    if not request.documents or not request.questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions' field in the request.")

    try:
        chunks = process_document(request.documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not process document from URL.")

        # Create embeddings for all chunks via OpenAI API
        chunk_embeddings = get_embeddings(chunks, client)

        all_answers = []
        for question in request.questions:
            # Create embedding for the question
            question_embedding = get_embeddings([question], client)[0]
            
            # Perform semantic search
            similarities = [np.dot(question_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
            top_k = 5
            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
            context = "\n\n---\n\n".join([chunks[i] for i in top_indices])
            
            # Generate answer with GPT-4 using the retrieved context
            answer = generate_answer_with_gpt4(question, context, client)
            all_answers.append(answer)
            
        return HackRxResponse(answers=all_answers)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(router)

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "model_info": "This endpoint uses OpenAI for embeddings and GPT-4 for generation."}
