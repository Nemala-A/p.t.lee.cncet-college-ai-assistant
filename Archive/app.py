
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "college_info.txt"
STATIC_DIR = BASE_DIR / "static"

# Create FastAPI application
app = FastAPI(title="College AI Assistant")

# Serve frontend files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


# Load college information
text = DATA_FILE.read_text(encoding="utf-8")

chunks = [
    section.strip()
    for section in text.split("\n\n")
    if section.strip()
]

# Load embedding model
print("Loading AI embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create vector search index
embeddings = model.encode(chunks, normalize_embeddings=True)
embeddings = np.asarray(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("College AI Assistant is ready!")


# Request format from website
class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    # Correct common spelling mistakes
    question_lower = question.lower()

    typo_corrections = {
        "principle": "principal",
        "principel": "principal",
        "pricipal": "principal",
    }

    for wrong, correct in typo_corrections.items():
        question_lower = question_lower.replace(wrong, correct)

    question = question_lower

    # Find relevant college information
    query_vector = model.encode(
        [question],
        normalize_embeddings=True
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    scores, results = index.search(
        query_vector,
        k=min(3, len(chunks))
    )

    retrieved_chunks = [
        chunks[idx]
        for idx in results[0]
        if idx >= 0
    ]

    # Include courses section for course questions
    course_keywords = [
        "course", "courses", "enna course",
        "enna courses", "padippu", "padippugal"
    ]

    if any(word in question for word in course_keywords):
        for chunk in chunks:
            if "COURSES LISTED ON WEBSITE" in chunk.upper():
                if chunk not in retrieved_chunks:
                    retrieved_chunks.append(chunk)
                break

    # Include principal section for principal questions
    principal_keywords = [
        "principal", "principel", "pricipal"
    ]

    if any(word in question for word in principal_keywords):
        for chunk in chunks:
            if "PRINCIPAL" in chunk.upper():
                if chunk not in retrieved_chunks:
                    retrieved_chunks.append(chunk)
                break

    
    # Include courses section for course questions
    course_keywords = [
        "course", "courses", "enna course",
        "enna courses", "padippu", "padippugal"
    ]

    if any(word in question for word in course_keywords):
        for chunk in chunks:
            if "COURSES LISTED ON WEBSITE" in chunk.upper():
                if chunk not in retrieved_chunks:
                    retrieved_chunks.append(chunk)
                break

    # Include principal section for principal questions
    principal_keywords = [
        "principal", "principel", "pricipal", "principle"
    ]

    if any(word in question for word in principal_keywords):
        for chunk in chunks:
            if "PRINCIPAL" in chunk.upper():
                if chunk not in retrieved_chunks:
                    retrieved_chunks.insert(0, chunk)

    # Include contact details for contact questions
    contact_keywords = [
        "contact", "phone", "telephone", "email",
        "mobile", "number", "reach", "address"
    ]

    if any(word in question for word in contact_keywords):
        for chunk in chunks:
            if "PHONE:" in chunk.upper() or "PRINCIPAL EMAIL" in chunk.upper():
                if chunk not in retrieved_chunks:
                    retrieved_chunks.insert(0, chunk)
                break

    # Join retrieved information correctly
    context = "\n\n".join(retrieved_chunks)


    # Ask Gemma to answer using retrieved information
    prompt = f"""
You are the assistant for P.T. Lee Chengalvaraya Naicker
College of Engineering & Technology.

Use only the college information below.
Never invent college details.

Answer in the same language as the student's question.
For Tamil questions, answer in natural Tamil.
For Tanglish questions, answer in natural Tanglish.
For English questions, answer in simple English.
Keep the answer short and friendly.

If the information does not contain the answer, say so.

College information:
{context}

Student question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        answer = response.json()["response"]

        return {"answer": answer}

    except requests.RequestException as error:
        print("Ollama connection error:", error)
        raise HTTPException(
            status_code=503,
            detail="AI service is unavailable. Please try again."
        )
