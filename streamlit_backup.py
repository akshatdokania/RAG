import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# Load environment variables
load_dotenv()

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
    You are a teaching assistant for a Data Science course. Your role is to assist students with their queries by leveraging the provided context.
    Use the retrieved information below to formulate clear, concise, and accurate responses.
    Ensure the answer directly addresses the query while remaining succinct.
    Context: {context}
    Question: {question}
"""

custom_rag_prompt = PromptTemplate.from_template(template)

# Define RAG chain
rag_chain = LLMChain(
    prompt=custom_rag_prompt,
    llm=llm,
    output_parser=StrOutputParser(),
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Display chat history
st.markdown("<div style='margin-bottom: 100px;'>", unsafe_allow_html=True)  # Add spacing for chatbox
for message in st.session_state.messages:
    role = message["role"].capitalize()
    with st.chat_message(role):
        st.markdown(message["content"])
st.markdown("</div>", unsafe_allow_html=True)

# Function to clear text
def clear_text():
    st.session_state.user_input = ""

# Fixed position for input box and submit button
with st.container():
    col1, col2 = st.columns([9, 1])
    with col1:
        user_input = st.text_input(
            "Ask a policy-related question:",
            value=st.session_state["user_input"],
            label_visibility="collapsed",
            key="user_input_box",
            on_change=clear_text
        )
    with col2:
        submit_clicked = st.button("Submit")

if submit_clicked and user_input:
    st.session_state["user_input"] = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # Get response from RAG chain
        context = retriever.get_relevant_documents(user_input)
        response = rag_chain.run({"context": context, "question": user_input})

        

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        with st.chat_message("assistant"):
            st.markdown(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Clear input after submission
    st.session_state["user_input"] = ""
