import os
import warnings
import logging
from typing import List, Optional

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever


# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables.")
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 
import os
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

from langchain.vectorstores import Pinecone
from pinecone import Pinecone,ServerlessSpec
pc=Pinecone(api_key=PINECONE_API_KEY)
index_name = "medibot"
index=pc.Index(index_name)

try:
    from langchain_pinecone import PineconeVectorStore
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    logger.info("Vector store loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load vector store: {e}")
    raise

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# Define prompt template
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Act as a chatbot.
    Use the following context to answer the question directly and concisely in detail.
    If question is from outside medical fiend jsut say "Please ask amedical Query".
    Your answer should be formal and simple string.
    Context:
    {context}

    Question: {question}
    Answer:
    """
)

# Initialize memory buffer for conversation
memory = ConversationBufferMemory(
    k=10,
    return_messages=True,
    memory_key="chat_history",
    input_key="question"
)

# Initialize language model
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7
)



# Initialize RAG chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
)

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response schemas
class ChatRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        history = payload.history or []

        # Use asyncio.to_thread to run blocking rag_chain.invoke without blocking event loop
        result = await asyncio.to_thread(
            rag_chain.invoke,
            {"question": payload.question, "history": history}
        )

        return ChatResponse(answer=result["answer"])

    except Exception as e:
        logger.exception("Error in /chat endpoint:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Chatbot API. Use the /chat endpoint to interact with the chatbot."}



