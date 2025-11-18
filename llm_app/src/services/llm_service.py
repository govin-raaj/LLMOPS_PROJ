from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from config.config import Config
from src.logger import logging


class LLMService:
    def __init__(self, vector_store):
        self.llm = HuggingFaceEndpoint(
            repo_id="google/gemma-2-2b-it",
            task="text-generation",
            huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_KEY,
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.chain=ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.vectorstore.as_retriever(),
            memory=self.memory, 
        )



    def get_response(self, question):
        try:
            response=self.chain({"question": question})
            return response['answer']
        
        except Exception as e:
            logging.error(f"Error in LLMService get_response: {e}")
            return "Error generating response."