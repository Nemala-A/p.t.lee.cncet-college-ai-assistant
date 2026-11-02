
from pathlib import Path
import os
import asyncio
import edge_tts

from faster_whisper import WhisperModel
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# FASTAPI APPLICATION
# ==============================

app = FastAPI(title="College AI Assistant")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


# ==============================
# LOAD ALL COLLEGE TEXT FILES
# ==============================

def load_college_data():
    text_files = sorted(DATA_DIR.glob("*.txt"))

    if not text_files:
        raise RuntimeError(
            f"No .txt files found in {DATA_DIR}. "
            "Please add college information files."
        )

    chunks = []

    for file_path in text_files:
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        # Split text into paragraphs
        paragraphs = [
            p.strip()
            for p in content.split("\n\n")
            if p.strip()
        ]

        current_chunk = ""

        for paragraph in paragraphs:
            # Split unusually long paragraphs
            if len(paragraph) > 1000:
                if current_chunk:
                    chunks.append(
                        f"[Source: {file_path.name}]\n{current_chunk}"
                    )
                    current_chunk = ""

                for start in range(0, len(paragraph), 1000):
                    part = paragraph[start:start + 1000]
                    chunks.append(
                        f"[Source: {file_path.name}]\n{part}"
                    )
                continue

            if len(current_chunk) + len(paragraph) + 2 > 1000:
                if current_chunk:
                    chunks.append(
                        f"[Source: {file_path.name}]\n{current_chunk}"
                    )
                current_chunk = paragraph
            else:
                current_chunk = (
                    current_chunk + "\n\n" + paragraph
                    if current_chunk
                    else paragraph
                )

        if current_chunk:
            chunks.append(
                f"[Source: {file_path.name}]\n{current_chunk}"
            )

    if not chunks:
        raise RuntimeError(
            "College text files are empty. Please check the data folder."
        )

    print(f"Loaded {len(text_files)} text files.")
    print(f"Created {len(chunks)} searchable chunks.")

    return chunks


print("Loading college information...")
chunks = load_college_data()


# ==============================
# LOAD EMBEDDING MODEL
# ==============================

print("Loading AI embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
# ==============================
# LOAD SPEECH-TO-TEXT MODEL
# ==============================

print("Loading Whisper voice model...")

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper voice model is ready!")

# ==============================
# CREATE FAISS SEARCH INDEX
# ==============================

print("Creating FAISS search index...")

embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    show_progress_bar=False
)

embeddings = np.asarray(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("College AI Assistant is ready!")


# ==============================
# CHAT REQUEST FORMAT
# ==============================

class ChatRequest(BaseModel):
    question: str


# ==============================
# KEYWORD HELPERS
# ==============================

def get_matching_chunks(keywords, limit=4):
    matches = []

    for chunk in chunks:
        chunk_lower = chunk.lower()

        if any(keyword in chunk_lower for keyword in keywords):
            matches.append(chunk)

        if len(matches) >= limit:
            break

    return matches


# ==============================
# CHAT API
# ==============================

@app.post("/api/chat")
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    question_lower = question.lower()

    # Correct common spelling mistakes
    typo_corrections = {
        "principel": "principal",
        "pricipal": "principal",
        "principle": "principal",
        "faculity": "faculty",
        "faculties": "faculty",
        "placment": "placement",
        "placemnt": "placement",
    }

    for wrong, correct in typo_corrections.items():
        question_lower = question_lower.replace(wrong, correct)

    question = question_lower

    # Answer common principal identity questions directly
    principal_intent = any(
        word in question
        for word in [
            "who", "name", "yaaru", "yaru", "yaar",
            "peru", "who is", "tell me"
        ]
    )

    if "principal" in question and principal_intent:
        return {
            "answer": (
                "The principal of P.T. Lee Chengalvaraya Naicker "
                "College of Engineering and Technology is "
                "Dr. P. Palanisamy, M.E., Ph.D."
            )
        }

    # Find semantically relevant chunks using FAISS
    query_vector = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32"
    )

    number_to_search = min(5, len(chunks))

    scores, results = index.search(
        query_vector,
        number_to_search
    )

    retrieved_chunks = []

    for idx in results[0]:
        if idx >= 0:
            retrieved_chunks.append(chunks[idx])

    # Add extra information based on question topic
    topic_keywords = {
        "course": [
            "course", "courses", "department", "departments",
            "padippu", "padippugal", "engineering branches"
        ],
        "faculty": [
            "faculty", "teacher", "teachers", "professor",
            "professors", "staff", "lecturer", "lecturers"
        ],
        "placement": [
            "placement", "placements", "recruiter", "recruiters",
            "salary package", "companies", "qspiders",
            "aram control", "zinnov"
        ],
        "contact": [
            "contact", "phone", "telephone", "email",
            "mobile", "address", "reach"
        ],
        "facility": [
            "facility", "facilities", "library", "laboratory",
            "laboratories", "hostel", "transport", "infrastructure"
        ],
        "admission": [
            "admission", "admissions", "tnea", "eligibility",
            "fees", "fee", "counselling"
        ],
        "principal": [
            "principal", "principel", "pricipal"
        ],
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in question for keyword in keywords):

            if topic == "course":
                extra_chunks = get_matching_chunks([
                    "courses offered",
                    "courses listed",
                    "engineering departments",
                    "computer science and engineering",
                    "artificial intelligence and data science",
                ])

            elif topic == "faculty":
                extra_chunks = get_matching_chunks([
                    "faculty directory",
                    "faculty information",
                    "assistant professor",
                    "associate professor",
                    "professor",
                    "computer science and engineering",
                    "mechanical engineering",
                ])

            elif topic == "placement":
                extra_chunks = get_matching_chunks([
                    "placement",
                    "recruiting companies",
                    "qspiders",
                    "aram control",
                    "zinnov",
                ])

            elif topic == "contact":
                extra_chunks = get_matching_chunks([
                    "contact details",
                    "phone:",
                    "principal email",
                    "office email",
                    "address:",
                ])

            elif topic == "facility":
                extra_chunks = get_matching_chunks([
                    "college facilities",
                    "laboratories",
                    "library",
                    "transport",
                    "campus facilities",
                ])

            elif topic == "admission":
                extra_chunks = get_matching_chunks([
                    "admissions",
                    "tnea",
                    "admission information",
                    "eligibility",
                    "fee details",
                ])

            else:
                extra_chunks = get_matching_chunks([
                    "principal",
                    "dr. p. palanisamy",
                ])

            for chunk in extra_chunks:
                if chunk not in retrieved_chunks:
                    retrieved_chunks.append(chunk)

    # Prepare context for Gemma
    context = "\n\n---\n\n".join(retrieved_chunks)

    if not context.strip():
        context = "No matching college information was found."

    # Ask Gemma to answer using retrieved college data
    prompt = f"""
You are the AI assistant for P.T. Lee Chengalvaraya Naicker
College of Engineering and Technology.

RULES:
1. Use only the college information provided below.
2. Do not invent names, email addresses, phone numbers,
   fees, statistics, or other college details.
3. Answer in the same language as the student's question.
4. For Tamil questions, answer in natural Tamil.
5. For Tanglish questions, answer in natural Tanglish.
6. For English questions, use simple English.
7. Keep answers clear and friendly.
8. If the information is missing, say it is not available
   in the current college data.
9. When asked about faculty, use the faculty information
   present in the supplied context.
10. Do not claim a company recruits students unless the
    supplied context supports that information.

COLLEGE INFORMATION:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    # Call local Ollama model
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

        answer = response.json().get("response", "").strip()

        if not answer:
            raise HTTPException(
                status_code=503,
                detail="The AI returned an empty answer. Please try again."
            )

        return {"answer": answer}

    except requests.RequestException as error:
        print("Ollama connection error:", error)

        raise HTTPException(
            status_code=503,
            detail=(
                "AI service is unavailable. Check whether Ollama "
                "is running and the gemma3:4b model is installed."
            )
        )
 # ==============================
# VOICE INPUT API
# ==============================

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):

    audio_path = DATA_DIR / "voice_input.webm"

    try:
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="No audio received."
            )

        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_bytes)

        segments, info = whisper_model.transcribe(
            str(audio_path),
            beam_size=5
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        return {
            "text": transcript,
            "language": info.language
        }

    finally:
        await audio.close()
        if audio_path.exists():
            audio_path.unlink()