# main.py
import os
import requests
import numpy as np
from fastapi import FastAPI, HTTPException, Header
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

# --- FastAPI App ---
app = FastAPI()

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
    prompt = f"""
    You are an expert Q&A system. Your answers must be based *only* on the provided context.
    If the answer cannot be found in the context, state that clearly and concisely. Don't mention that you have read the document, it should be a direct one liner answer.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    try:
        # Using gpt-4o as it's powerful and cost-effective
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        if "authentication" in str(e).lower():
            raise HTTPException(status_code=401, detail="Authentication failed with OpenAI. Check your API key.")
        raise HTTPException(status_code=500, detail="Failed to generate answer from OpenAI.")

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(
    request: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    # 1. Validate and extract the OpenAI API key from the Bearer token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    api_key = authorization.split(" ")[1]
    if not api_key:
        raise HTTPException(status_code=401, detail="Bearer token is empty.")

    try:
        # 2. Initialize the OpenAI client for this specific request
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize OpenAI client: {e}")

    if not request.documents or not request.questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions' field in the request.")

    try:
        chunks = process_document(request.documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not process document from URL.")

        # 3. Create embeddings for all chunks via OpenAI API
        chunk_embeddings = get_embeddings(chunks, client)

        all_answers = []
        for question in request.questions:
            # 4. Create embedding for the question
            question_embedding = get_embeddings([question], client)[0]
            
            # 5. Perform semantic search
            similarities = [np.dot(question_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
            top_k = 5
            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
            context = "\n\n---\n\n".join([chunks[i] for i in top_indices])
            
            # 6. Generate answer with GPT-4 using the retrieved context
            answer = generate_answer_with_gpt4(question, context, client)
            all_answers.append(answer)
            
        return HackRxResponse(answers=all_answers)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "model": "OpenAI GPT-4"}
