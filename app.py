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
from langsmith import traceable
from pix2text import Pix2Text
from pdf2image import convert_from_path
import PyPDF2
from pix2text import Pix2Text
import tempfile
import tempfile
import re


# Load environment variables
load_dotenv()


os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Tutor-deployed"

# Load API keys from secrets.toml
# Correct key names based on secrets.toml
openai_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
langsmith_key = st.secrets["api_keys"]["LANGSMITH_API_KEY"]

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LANGSMITH_API_KEY"] = langsmith_key  # Set the LangSmith API key

# Load prompts from secrets.toml
contextualize_q_system_prompt = st.secrets["ds120_prompts"]["contextualize_q_system_prompt"]
qa_prompt_template = st.secrets["ds120_prompts"]["qa_prompt_template"]





@traceable
def invoke_rag_chain(prompt: str, chat_history: list):
    response = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
    return response



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



def preprocess_latex_in_response(response):
    """
    Detect unformatted LaTeX-style expressions and process them,
    while skipping already valid LaTeX math and avoiding over-wrapping.
    """
    # Regex to detect unformatted LaTeX-style expressions (e.g., (theta), (x^i))
    # Avoids already valid `$ ... $` or `$$ ... $$`
    # equation_pattern = r"(?<!\$)\(([\\a-zA-Z0-9_^,]+)\)(?!\$)"  # Detect math in parentheses

    # # Skip already valid inline or block LaTeX
    # valid_math_pattern = r"(\$\$.*?\$\$|\$.*?\$)"  # Matches valid `$ ... $` or `$$ ... $$`
    
    # # Placeholder for valid LaTeX to prevent modification
    # valid_math_placeholders = []
    # response_with_placeholders = response

    # # Replace valid math blocks with placeholders
    # for match in re.finditer(valid_math_pattern, response):
    #     placeholder = f"__VALID_MATH_{len(valid_math_placeholders)}__"
    #     valid_math_placeholders.append(match.group(0))
    #     response_with_placeholders = response_with_placeholders.replace(match.group(0), placeholder, 1)

    # # Process remaining parts of the response for unformatted math
    # def format_equation(match):
    #     equation = match.group(1)  # Extract content inside parentheses
    #     return f"$$ {equation} $$"  # Wrap in $$ for block rendering

    # processed_response = re.sub(equation_pattern, format_equation, response_with_placeholders)

    # # Restore valid math blocks from placeholders
    # for i, valid_math in enumerate(valid_math_placeholders):
    #     processed_response = processed_response.replace(f"__VALID_MATH_{i}__", valid_math, 1)

    return response

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
llm = ChatOpenAI(model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"])

contextualize_q_system_prompt = st.secrets["ds120_prompts"]["contextualize_q_system_prompt"]
qa_prompt_template = st.secrets["ds120_prompts"]["qa_prompt_template"]

# Define contextualize prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Define the question-answering system prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
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
def process_uploaded_file(file):
    try:
        extracted_text = ""

        if file.type == "application/pdf":
            # Handle PDF files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.getvalue())
                temp_file.flush()

                # Try extracting text directly using PyPDF2
                try:
                    with open(temp_file.name, "rb") as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        for page in reader.pages:
                            extracted_text += page.extract_text() or ""  # Append text if available
                except Exception as e:
                    print(f"Text extraction from PDF failed: {e}. Proceeding with image conversion.")

                # If no text was extracted or fallback is needed
                if not extracted_text.strip():
                    pages = convert_from_path(temp_file.name, dpi=300)
                    p2t = Pix2Text.from_config(device="cpu")
                    for i, page in enumerate(pages):
                        temp_image_path = f"/tmp/page_{i + 1}.jpg"  # Save temporary images
                        page.save(temp_image_path, "JPEG")
                        page_text = p2t.recognize(temp_image_path)
                        extracted_text += page_text + "\n"

        elif file.type.startswith("image/"):
            # Handle image files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(file.getvalue())
                temp_file.flush()

                # Extract text from the image using Pix2Text
                p2t = Pix2Text.from_config(device="cpu")
                extracted_text = p2t.recognize(temp_file.name)

        else:
            st.error("Unsupported file type. Please upload an image or a PDF.")
            return None

        if not extracted_text.strip():
            st.warning("No text could be extracted from the file.")
            return None

        return extracted_text.strip()

    except Exception as e:
        st.error(f"Failed to process the file: {str(e)}")
        return None

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
    .stElementContainer.element-container:empty {
    display: none !important; /* Completely remove the element visually */
}
    </style>
    """,
    unsafe_allow_html=True,
)
     

st.markdown(
    """
<style>
/* Explicitly isolate the enclosing box */
[data-testid="stHorizontalBlock"] {
    position: fixed;
    bottom: 10px; /* Keep glued to the bottom */
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    z-index: 1000; /* Ensure visibility above other elements */
    background: rgba(13,17,24,255);
    color: white;
    padding: 0; /* Reset all padding */
    padding-bottom: 10px;
    padding-left: 10px;
    border-radius: 15px;
    display: flex;
    flex-direction: column;
    gap: 0px; /* Remove gap between file uploader and input box */
    justify-content: flex-start;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Ensure children are isolated inside the enclosing block */
[data-testid="stHorizontalBlock"] > * {
    margin: 0;
    padding: 0;
}

/* Optional: Debugging overflow issues (ensures children don't escape) */
[data-testid="stHorizontalBlock"] {
    overflow: hidden;
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
/* Chat message - Always white background without borders */
[data-testid="stChatMessage"] {
    background: #ffffff !important; /* White background */
    color: #000000 !important; /* Black text for readability */
    padding: 10px; 
    border-radius: 10px; /* Rounded corners */
    border: none !important; /* No borders */
    box-shadow: none !important; /* No shadows */
}

/* User message with white background in light mode */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdownContainer {
    background-color: #ffffff; /* White background */
    color: #000000; /* Black text */
    border-radius: 10px 10px 0px 10px; /* Rounded corners */
    padding: 10px; 
    border: none !important; /* No borders */
    box-shadow: none !important; /* No shadows */
}

/* Assistant message with white background in light mode */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdownContainer {
    background-color: #ffffff; /* White background */
    color: #000000; /* Black text */
    border-radius: 10px 10px 10px 0px; /* Rounded corners */
    padding: 10px; 
    border: none !important; /* No borders */
    box-shadow: none !important; /* No shadows */
}

/* Input box and file uploader container - Light mode */
[data-testid="stHorizontalBlock"] {
    position: fixed;
    bottom: 10px; /* Glued to the bottom */
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    z-index: 1000;
    background: #ffffff; /* White for light mode */
    color: #000000; /* Black font color */
    padding: 15px; /* Extra padding for a clean look */
    border-radius: 15px; /* Rounded corners */
    border: none !important; /* No borders */
    box-shadow: none !important; /* No shadows */
    transition: background-color 0.3s ease, color 0.3s ease; /* Smooth transition */
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #121212; /* Dark background */
        color: #ffffff; /* Light text */
    }

    /* Chat message box - Dark mode */
    [data-testid="stChatMessage"] {
        background: rgba(13, 17, 24, 255) !important; /* Dark background */
        color: #ffffff !important; /* White text for readability */
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdownContainer {
        background-color: rgba(13, 17, 24, 255) !important; /* Dark background for user messages */
        color: #ffffff; /* White text */
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdownContainer {
        background-color: rgba(13, 17, 24, 255) !important; /* Dark background for assistant messages */
        color: #ffffff; /* White text */
    }

    /* Horizontal block - Dark mode */
    [data-testid="stHorizontalBlock"] {
        background: rgba(13, 17, 24, 255); /* Dark background */
        color: #ffffff; /* White text */
        border: none !important; /* No borders */
        box-shadow: none !important; /* No shadows */
    }
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

    # File uploader logic
    with col1:
        st.markdown('<div class="stFileUploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(label_visibility="collapsed", label="Upload a file (image or PDF)", key="file_uploader",type = ['png','pdf'])

        extracted_content = ""
        if uploaded_file:
            extracted_content = process_uploaded_file(uploaded_file)


    # Chat input logic
    # Chat input logic
    with col2:
        st.markdown('<div class="custom-col">', unsafe_allow_html=True)
        if prompt := st.chat_input("Type your question here..."):
            # Check if an image is uploaded and extract content
            if extracted_content:
                # Append "Attachment" to the user's input for UI display
                ui_display_prompt = f"{prompt} \n\n[Attachment]"
                # Append the extracted content to the prompt for LLM
                prompt = f"{prompt}\n\n{extracted_content}"
            else:
                # No image attached, use the prompt as is
                ui_display_prompt = prompt

            # Add user input (with "Attachment" if applicable) to the message history
            st.session_state.messages.append({"role": "user", "content": ui_display_prompt})
            with chat_messages:
                st.chat_message("user").write(ui_display_prompt)
                # Use a placeholder for "Thinking..."
                thinking_placeholder = st.empty()  # Create a placeholder dynamically
                with thinking_placeholder:
                    st.chat_message("assistant").write("Thinking...")  # Display "Thinking..."

            # Generate response for the current query
            try:
                # Get response from RAG chain
                response = invoke_rag_chain(prompt, st.session_state.chat_history)
                response_text = response["answer"]  # Fix LaTeX formatting in the response

                # Clear the "Thinking..." placeholder before showing the response
                thinking_placeholder.empty()

                # Update chat history
                st.session_state.chat_history = update_chat_history(
                    st.session_state.chat_history,
                    prompt,
                    response_text
                )

                # Add assistant response to the message history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Display the assistant's response
                with chat_messages:
                    st.chat_message("assistant").write(response_text)

            except Exception as e:
                # Handle errors gracefully
                error_message = f"An error occurred: {str(e)}"

                # Clear the "Thinking..." placeholder before showing the error message
                thinking_placeholder.empty()

                # Add error message to the message history
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with chat_messages:
                    st.chat_message("assistant").write(error_message)
