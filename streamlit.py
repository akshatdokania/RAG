import os
import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from pix2text import Pix2Text
import tempfile
import re


os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"  # Replace with your actual API key
os.environ["LANGSMITH_PROJECT"] = "TutorBot"

@traceable
def invoke_rag_chain(prompt: str, chat_history: list):
    response = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
    return response

def fix_latex_format(response_text):
    # Correct single dollar signs wrapping inline math
    response_text = re.sub(r'(?<!\\)\$(.+?)\$', r'$\1$', response_text)

    # Correct double dollar signs wrapping block-level math
    response_text = re.sub(r'(?<!\\)\$\$(.+?)\$\$', r'$$\1$$', response_text)

    # Replace math within parentheses ( ... ) with inline math delimiters $ ... $
    response_text = re.sub(r'\(\s*\\([a-zA-Z]+.*?\))\s*\)', r'$\1$', response_text)

    return response_text







def sanitize_latex(response_text):
    response_text = response_text.replace("\\", "\\\\")  # Double all backslashes
    response_text = response_text.replace("_", "\\_")   # Escape underscores
    response_text = response_text.replace("&", "\\&")   # Escape ampersands
    response_text = response_text.replace("$", "\\$")   # Escape plain dollar signs
    return response_text


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

template_assignment = """
Given this question, elaborate and explain it like explaining to a 5 year old.
"""
# Define custom prompt template
template = """
       You are a knowledgeable and empathetic teaching assistant for the DS-120 Data Science course, designed to teach fundamental concepts of Data Science. Your primary role is to assist students with queries strictly within the course scope.
            ### Guidelines:

            1. Scope of Assistance:
            
            - Focus exclusively on fundamental concepts of Data Science covered in DS-120(limited to the scope of the context). This course is a foundational course, no coding knowledge is to be imparted.
            - For questions that require solving, in no circumstance can you solve those questions, as your role is just to improve conceptual knowledge. Consider giving a clearer understanding of the question, or explaining the underlying concepts of the question, but don not solve and give final answer. 
            - For queries beyond the scope, politely indicate that this a bot trained to deal with this subject's queries only.
            - For trivial questions, consider replying in a way that the question is answered without harming the sentiments of the user.


            2. Mathematical and Special Characters:
            
            - Inline mathematical equations must be wrapped with single dollar signs, e.g., $a^2 + b^2 = c^2$.
                    - Block-level equations must be wrapped with double dollar signs($$), e.g., $$\int_a^b f(x) dx$$.
                    - Escape any dollar signs ($) used in plain text by prefixing them with a backslash (\\$).
                    - Use valid LaTeX syntax for mathematical expressions.
                    - Make sure that the mathematical equations are properly matched and formatted mathematical delimiters, particularly the single dollar signs $ and parentheses.


            3. Problem-Solving Approach:
            - Direct Responses: For clear, straightforward questions, provide concise answers within the scope.
            - Step-by-Step Solutions: For complex queries, break them into smaller parts and answer each step logically.
            - Start with simple explanations and gradually include technical details.

            4. Leveraging Context:
            - Use context provided in the query to supplement your answer.
            - If context is insufficient, ask clarifying questions.

            5. Follow Up Questions: 
            - If there is a followup question, consider explaining the concept in greater detail to ensure clarity is provided

            5. Clarity and Readability:
            - Use simple language and examples for better understanding.
            - Format responses with bullet points, numbered lists, and concise paragraphs.
            - The equations should be in a latex friendly format.

            6. Tone and Presentation**:
            - Maintain a friendly, professional, and encouraging tone.
            - Avoid referring to external datasets or corpora. Present yourself as specifically trained for DS-120.

        Context: {context}  
        Question: {input}

"""

# Define the system prompt
# contextualize_q_system_prompt = (
#     "Given a chat history and the latest user question "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, "
#     "just reformulate it if needed and otherwise return it as is."
# )

contextualize_q_system_prompt = (
    "Given a chat history consisting of all the questions user sent, and the latest user question "
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



# Function to process uploaded image and extract text
def process_uploaded_image(image):
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
            temp_file.write(image.getvalue())
            temp_file.flush()

            # Initialize Pix2Text
            p2t = Pix2Text.from_config(device="cpu")

            # Extract content from the image
            recognized_content = p2t.recognize(temp_file.name)
            return recognized_content
    except Exception as e:
        st.error(f"Failed to process the image: {str(e)}")
        return ""

# Define a scrollable main area for chat history and fixed bottom input area
# chat_container = st.container(height = 500)
# input_container = st.container()

# Display chat messages in the fixed top area
with st.container(height = 600):
    chat_messages = st.container()
    with chat_messages:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

st.markdown(
    """
    <style>
    /* Target and remove borders from the chat history container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important; /* Completely removes the border */
        box-shadow: none !important; /* Removes any shadow that might resemble a border */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Make the horizontal block fixed with custom width and centered */
    [data-testid="stHorizontalBlock"] {
        position: fixed;
        bottom: 20px; /* Add gap from the bottom */
        left: 50%; /* Center the block horizontally */
        transform: translateX(-50%); /* Adjust positioning to center */
        width: 80%; /* Set custom width (80% of the browser width) */
        z-index: 1000; /* Ensure visibility above other elements */
        background: rgba(13,17,24,255); /* Custom background color */
        color: white; /* Set text color for contrast */
        padding: 20px; /* Inner spacing */
        border-radius: 15px; /* Rounded corners */
        display: flex; /* Flex layout for child elements */
        gap: 20px; /* Space between child elements */
        justify-content: space-between; /* Evenly space child elements */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Optional: Add shadow for depth */
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    """
    <style>
    /* Set the same background color for both user and AI chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(13,17,24,255) !important; /* Apply the same background color */
        color: white !important; /* Ensure text is readable against the dark background */
        border: none !important; /* Optional: Remove borders if any */
        box-shadow: none !important; /* Optional: Remove shadows */
        padding: 10px; /* Optional: Add some inner spacing for better appearance */
        border-radius: 10px
        ; /* Optional: Add rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)









st.markdown(
    """
    <style>
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%; /* Make it span the full width of the screen */
        padding: 10px; /* Optional: Add some padding */
        /* Use flexbox for layout */
        gap: 0px; /* Add space between items */
        align-items: center; /* Vertically align items */
        
    }
    .custom-col {
       /* Adjust the height as needed */
        
        /* padding: 10px; */
    }
    
    [data-testid='stFileUploader'] {
        width: max-content;
        
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
        
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
        
    }

    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* Align user messages and avatars to the right */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        justify-content: flex-end; /* Align the container to the right */
        text-align: right; /* Ensure text is aligned right */
        flex-direction: row-reverse; /* Flip the order of avatar and message */
    }

    /* User avatar styling */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
        margin-left: 10px; /* Add spacing between avatar and message */
        margin-right: 0; /* Remove spacing on the right */
    }

    /* User message styling */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdownContainer {
        background-color: #1E90FF; /* Light blue background */
        color: white; /* White text color */
        border-radius: 10px 10px 0px 10px; /* Rounded corners */
        padding: 10px; /* Add padding */
    }

    /* Assistant message styling remains default */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        justify-content: flex-start; /* Keep assistant messages on the left */
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdownContainer {
        background-color: #333; /* Default dark background */
        color: white; /* White text color */
        border-radius: 10px 10px 10px 0px; /* Rounded corners */
        padding: 10px; /* Add padding */
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Fixed bottom layout for file upload and chat input
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 10])
    with col1:
        st.markdown('<div class="stFileUploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(label_visibility="collapsed", label="Upload an image", key="file_uploader")

        extracted_content = ""
        if uploaded_file:
            extracted_content = process_uploaded_image(uploaded_file)

    with col2:
    # Chat input logic
        st.markdown('<div class="custom-col">', unsafe_allow_html=True)
        if prompt := st.chat_input("Type your question here..."):
            # Append extracted content to the prompt if available
            if extracted_content:
                prompt = f"{prompt} \n\n {extracted_content}"

            # Add user input to the message history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_messages:
                st.chat_message("user").write(prompt)

            try:
                # Get response from the RAG chain via the traced function
                response = invoke_rag_chain(prompt, st.session_state.chat_history)
                response["answer"] = fix_latex_format(response["answer"])  # Correct LaTeX formatting

                # Update chat history
                st.session_state.chat_history = update_chat_history(
                    st.session_state.chat_history,
                    prompt,
                    response["answer"]
                )

                # Add assistant response to messages
                assistant_message = response["answer"]
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                with chat_messages:
                    with st.chat_message("assistant"):
                        # Render assistant message with LaTeX-compatible Markdown
                        st.markdown(assistant_message, unsafe_allow_html=True)

            except Exception as e:
                # Handle errors gracefully
                error_message = f"An error occurred: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with chat_messages:
                    st.chat_message("assistant").write(error_message)





