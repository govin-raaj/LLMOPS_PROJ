from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config.config import Config
from src.logger import logging


class LLMService:
    def __init__(self, vector_store):

        # Correct HF endpoint wrapper usage
        llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            task="text-generation",
            huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_KEY,
            max_new_tokens=200,
        )
        self.model = ChatHuggingFace(llm=llm)

        self.retriever = vector_store.vector_store.as_retriever(search_kwargs={"k": 4})




    def get_response(self, question):

        template="""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use ten sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
        prompt=ChatPromptTemplate.from_template(template)
        output_parser=StrOutputParser()
        rag_chain = (
            {"context": self.retriever,  "question": RunnablePassthrough()}
                | prompt
                | self.model
                | output_parser
            )
        try:
            result = rag_chain.invoke(question)
            return result
        

        except Exception as e:
            logging.error(f"Error in LLMService get_response: {e}")
            return "Error generating response."
