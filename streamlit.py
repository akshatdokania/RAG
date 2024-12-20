import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize the message history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # Initialize the chat history

# Streamlit app setup
st.set_page_config(
    page_title="DS-120 Virtual Teaching Assistant Chatbot",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center;'>DS-120 Virtual Teaching Assistant Chatbot</h1>", unsafe_allow_html=True)

# Initialize SentenceTransformer model for embeddings
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# Load FAISS vector store
vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

# Define custom prompt template
template = """
       You are a teaching assistant for a Data Science course. Your role is to assist students with their queries by leveraging the provided context, while not helping them with assigment solutions. Always Ensure the response follows the below rules:
    
    1. Always analyse the metadata of the context chunks first .If in the file_name of metatadata it is mentioned "Do not answer", don't give the solution to anything related to that content because it is supposed to be a part of an assignment.
    2. If there is nothing such mentioned in the metadta, Always Properly handle mathematical expressions and special characters:
        - Always Inline mathematical equations must be wrapped with single dollar signs, e.g., `$a^2 + b^2 = c^2$`.
        - Always Block-level equations must be wrapped with double dollar signs($$), e.g., `$$\int_a^b f(x) dx$$`.
        - Always Escape any dollar signs (`$`) used in plain text by prefixing them with a backslash (`\$`).
        - Always Use valid LaTeX syntax for mathematical expressions.
        - Always Ensure Python code or inline formulas are enclosed within backticks, e.g., `` `[f""${{i}}"" for i in range(1,11)]` ``.
    3.  Directly addresses the query while remaining clear, concise, and accurate.   

    
    Never answer questions that are out of scope and do not relate directly to our data, politely respond that you have been trained to deal with questions only related to the subject DS-120.
    Also keep the answers restricted to the scope of the subject, do not explain beyond.

    Context: {context}
    Question: {input}
"""

# Define the system prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Define contextualize prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),  # Correct variable naming
        ("human", "{input}"),  # Updated to match the expected variable name
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Define the question-answering system prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),  # Correct variable naming
        ("human", "{input}"),  # Updated to match the expected variable name
    ]
)

# Create a question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to manage chat history and keep the latest 5 messages
def update_chat_history(chat_history, human_message, ai_message):
    chat_history.extend(
        [
            HumanMessage(content=human_message),
            AIMessage(content=ai_message),
        ]
    )
    # Keep only the last 5 messages
    if len(chat_history) > 5:
        chat_history = chat_history[-5:]
    return chat_history

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input logic
if prompt := st.chat_input("Type your question here..."):
    # Add user input to the message history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        # Get response from RAG chain
        response = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})

        # Update chat history
        st.session_state.chat_history = update_chat_history(
            st.session_state.chat_history,
            prompt,
            response["answer"]
        )

        # Add assistant response to messages
        assistant_message = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        st.chat_message("assistant").write(assistant_message)

    except Exception as e:
        # Handle errors gracefully
        error_message = f"An error occurred: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.chat_message("assistant").write(error_message)
