from Fastapi import FastAPI, request, jsonify
from models.vector_store import VectorStore
from services.storage_service import S3Storage
from services.llm_service import LLMService
from config import Config
import os
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from src.logger import logging


app = FastAPI()
vectore_store = VectorStore()
llm_service = LLMService(vectore_store)

@app.get("/")
def home():
    return "Welcome to the LLM Application!"

@app.post("/upload")
def upload_document():
    try:
        logging.info("Upload endpoint called.")

        if 'file' not in request.files:
            logging.error("No file part in the request.")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file.")
            return jsonify({"error": "No selected file"}), 400
        
        if not file.filename.endswith(('.txt', '.pdf')):
            logging.error(f"Unsupported file type: {file.filename}")
            return jsonify({"error": "Unsupported file type, only .txt and .pdf are supported."}), 400
        
        logging.info(f"Processing file: {file.filename}")

        try:
            text_chunks=process_document(file)
            logging.info(f"Document processed into {len(text_chunks)} chunks.")
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            return jsonify({"error": "Error processing document."}), 500
        
        #vector_store
        try:
            vectore_store.add_documents(text_chunks)
            logging.info("Documents added to vector store.")
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            return jsonify({"error": "Error adding documents to vector store."}), 500
        
        return jsonify({"message": "File uploaded and processed successfully."}), 200
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

        


def process_document(file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        file.save(temp_path)

        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file.filename.endswith('.txt'):
            loader = TextLoader(temp_path)
            documents = loader.load()
        else:
            raise ValueError("Unsupported file type")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        text_chunks = text_splitter.split_documents(documents)

        return text_chunks
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)


@app.post("/query")
def query():
    data=request.json
    if 'question' not in data:
        logging.error("No question provided in the request.")
        return jsonify({"error": "No question provided."}), 400
    
    try:
        response=llm_service.get_response(data['question'])
        logging.info("Response generated successfully.")
        return jsonify({"response": response}), 200
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({"error": "Error generating response."}), 500



