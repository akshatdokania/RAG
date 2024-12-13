import chainlit as cl
import os
import json
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

# Setup LangChain model and embeddings
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Load chunks from JSON
def load_documents_from_json(json_file="chunks.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)["chunks"]
    documents = [
        Document(page_content=chunk["content"], metadata={"id": chunk["id"]})
        for chunk in data if chunk["content"].strip()
    ]
    print(f"Loaded {len(documents)} documents from {json_file}")
    return documents

@cl.on_chat_start
async def start_chat():
    global rag_chain
    
    # Load documents and create vector store
    documents = load_documents_from_json()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Setup LLM and conversation chain
    llm = ChatOpenAI(model="gpt-4o-mini")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    system_prompt = (
        "You are a teaching assistant for a Data Science course. Your role is to assist students with their queries "
        "by leveraging the provided context. Use the retrieved information below to formulate clear, concise, and accurate responses."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        return_source_documents=True,
    )

    await cl.Message(content="Vector Store and RAG chain are ready! You can now ask questions.").send()

@cl.on_message
async def handle_question(message):
    global rag_chain
    if not rag_chain:
        await cl.Message(content="RAG pipeline is not initialized. Please wait.").send()
        return

    res = await rag_chain.acall(message.content)
    answer = res["answer"]
    await cl.Message(content=answer).send()
