import chainlit as cl
import os
import zipfile
from pdf2image import convert_from_path
from pix2text import Pix2Text
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import markdown
import json

# Global variables for vector store and RAG chain
vector_store = None
rag_chain = None
welcome_message = """Welcome! To get started:
1. Upload a ZIP file containing PDFs, or a single PDF or text file.
2. Ask questions based on the content of the file(s).
"""

# Paths for processing files
pdf_folder = "./pdfs"
output_folder = "./extracted_pages"
markdown_folder = "./output-markdown"
chunks_file = "chunks.json"

# Setup LangChain model and embeddings
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(markdown_folder, exist_ok=True)

# File processing function
def process_file(file):
    # Check file type
    if file.type == "application/zip":
        with zipfile.ZipFile(file.path, 'r') as zip_ref:
            zip_ref.extractall(pdf_folder)
        print(f"Extracted ZIP contents: {os.listdir(pdf_folder)}")
    elif file.type == "application/pdf":
        loader = PyPDFLoader(file.path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        pdf_path = os.path.join(pdf_folder, file.name)
        with open(pdf_path, "wb") as f:
            f.write(docs)
        print(f"PDF saved to {pdf_path}")

    # Process PDFs
    p2t = Pix2Text.from_config(device="cpu")
    for root, _, files in os.walk(pdf_folder):
        for file_name in files:
            if file_name.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file_name)
                print(f"Processing PDF: {pdf_path}")
                pages = convert_from_path(pdf_path, dpi=300)
                print(f"Extracted {len(pages)} pages from {file_name}")
                for i, page in enumerate(pages):
                    output_filename = f"{os.path.splitext(file_name)[0]}_page_{i + 1}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    page.save(output_path, "JPEG")
                    print(f"Saved image: {output_path}")

    # Extract text
    for filename in sorted(os.listdir(output_folder)):
        file_path = os.path.join(output_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Running OCR on {file_path}")
            recognized_content = p2t.recognize(file_path)
            print(f"OCR result: {recognized_content[:100]}")  # Print first 100 characters
            output_file = os.path.join(markdown_folder, f"{os.path.splitext(filename)[0]}.md")
            with open(output_file, "w", encoding="utf-8") as md_file:
                md_file.write(recognized_content)
                print(f"Markdown saved: {output_file}")

    # Chunk text
    chunks_with_metadata = []
    for md_filename in sorted(os.listdir(markdown_folder)):
        md_file_path = os.path.join(markdown_folder, md_filename)
        if md_filename.endswith(".md"):
            with open(md_file_path, "r", encoding="utf-8") as file:
                content = file.read()
            print(f"Chunking text for {md_filename}")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_text(content)
            print(f"Generated {len(chunks)} chunks for {md_filename}")
            for idx, chunk in enumerate(chunks):
                chunks_with_metadata.append({
                    "metadata": {"id": idx + 1, "file": md_filename},
                    "content": chunk
                })

    # Save chunks
    print(f"Saving chunks to {chunks_file}")
    with open(chunks_file, "w", encoding="utf-8") as json_file:
        json.dump({"chunks": [{"id": idx + 1, "content": chunk["content"]} for idx, chunk in enumerate(chunks_with_metadata)]}, json_file, indent=4)

    # Create documents
    documents = [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunks_with_metadata
        if chunk["content"].strip()
    ]
    print(f"Created {len(documents)} documents")
    return documents


@cl.on_chat_start
async def start_chat():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["application/zip", "application/pdf", "text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    documents = await cl.make_async(process_file)(file)

    msg.content = f"`{file.name}` processed. Now building Vector Store and RAG chain!"
    await msg.update()

    # Build vector store and RAG chain
    global vector_store, rag_chain
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    system_prompt = (
        "You are a teaching assistant for a Data Science course. Your role is to assist students with their queries by leveraging the provided context. "
        "Use the retrieved information below to formulate clear, concise, and accurate responses. Ensure the answer directly addresses the query while remaining succinct."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        return_source_documents=True,
    )

    msg.content = f"`Vector Store and RAG chain have been built. You can now ask questions!"
    await msg.update()


@cl.on_message
async def handle_question(message):
    global rag_chain
    if not rag_chain:
        await cl.Message(content="Please upload a file first.").send()
        return

    res = await rag_chain.acall(message.content)
    answer = res["answer"]

    await cl.Message(content=answer).send()
