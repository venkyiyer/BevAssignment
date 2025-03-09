from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain.vectorstores import Chroma
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import config
import os


app = FastAPI()
os.environ["OPENAI_API_KEY"] = ""

# Base model for text input
class TextInput(BaseModel):
    text: str

class Conversation:
    def __init__(self):
        self.hf_embedding = HuggingFaceEmbeddings(model_name = config.embedding_model_name)
        self.vector_store = Chroma(collection_name= config.name_of_collection, persist_directory=config.vector_store_path, embedding_function=self.hf_embedding)
        self.llm = OpenAI(temperature = config.llm_temperature)
        self.chat_history = []
    
    # List to track chat history
    def get_chat_history(self, input):
        res = []
        for human, ai in input:
            res.append(f"Human:{human}\nAI:{ai}")
            
        return "\n".join(res)

    # Open AI api to converse. Vector store used as a retreiver
    def set_bot(self):
        retreiver = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.bot_chat = ConversationalRetrievalChain.from_llm(self.llm, 
                                                              retreiver, 
                                                              return_source_documents=True,
                                                              get_chat_history = self.get_chat_history,
                                                              verbose=True)
    
    # Prompts
    def generate_prompt(self, question):
        if not self.chat_history:
            prompt = f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. \nQuestion: {question}\nContext: \nAnswer:"
        else:
            context_entries = [f"Question: {q}\nAnswer: {a}" for q, a in self.chat_history[-3:]]
            context = "\n\n".join(context_entries)
            prompt = f"Using the context provided by recent conversations, answer the new question in a concise and informative. Limit your answer to a maximum of three sentences.\n\nContext of recent conversations:\n{context}\n\nNew question: {question}\n\Answer:"
        
        return prompt

    # QnA 
    def ask_question(self, query):
        prompt = self.generate_prompt(query)
        result = self.bot_chat.invoke({"question": prompt, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))

        return result["answer"]

# Fast API Endpoint
@app.post("/start_chat")
def start_conversation(input_data: TextInput):
    try:
        user_text = input_data.text
        chat_bot = Conversation()
        chat_bot.set_bot()
        return chat_bot.ask_question(user_text)
    except Exception as e:
        raise HTTPException (status_code=500, detail= str(e))
        

