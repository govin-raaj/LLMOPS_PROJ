import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates

from model.vector_store import VectorStore
from services.llm_service import LLMService
from config.config import Config

from langchain_community.document_loaders  import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import logging


app = FastAPI(title="RAG LLM API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStore(Config.VECTOR_DB_PATH)
llm_service = LLMService(vector_store)


def process_document(uploaded_file: UploadFile):
    """Process PDF/TXT and return text chunks"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.file.read())

        if uploaded_file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.filename.endswith(".txt"):
            loader = TextLoader(temp_path)
        else:
            raise HTTPException(status_code=400,
                                detail="Unsupported file type (only .txt, .pdf allowed).")

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)
        return chunks

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logging.info(f"Processing upload: {file.filename}")

        chunks = process_document(file)
        vector_store.add_documents(chunks)

        return {"message": "File processed successfully.",
                "chunks_processed": len(chunks)}

    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(500, "Error processing file")



class QueryModel(BaseModel):
    question: str


@app.post("/query")
async def query_llm(data: QueryModel):
    try:
        response = llm_service.get_response(data.question)
        return {"response": response}

    except Exception as e:
        logging.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse) 
async def home(request: Request): 
    return templates.TemplateResponse("index.html", {"request": request})

# Run with: uvicorn main:app --host 0.0.0.0 --port 8080 --reload
