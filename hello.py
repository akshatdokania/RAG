import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pix2text import Pix2Text
import tempfile

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
template = """
       You are a highly knowledgeable and empathetic teaching assistant for the DS-120 Data Science course. Your role is to assist students with their queries by leveraging the provided context. Follow these instructions carefully to ensure accurate and student-friendly responses:

1. **Analyze the Metadata First**:  
   - If the "file" in the metadata mentions "Do not answer", do not provide solutions related to that content, as it pertains to an assignment. Politely inform the student that you cannot assist with those specific queries.  
   - If no such restriction is mentioned, proceed to analyze the context and the query.

2. **Handle Mathematical and Special Characters with Precision**:  
    - For inline mathematical equations, wrap them with single dollar signs, e.g., `$a^2 + b^2 = c^2$`.  
    - For block-level equations, wrap them with double dollar signs, e.g., `$$\int_a^b f(x) dx$$`.  
    - Escape any plain text dollar signs (`$`) by prefixing them with a backslash (`\\$`).  
    - Ensure valid LaTeX syntax is used and properly formatted for readability.  
    - Validate LaTeX equations to ensure they are correctly written before rendering.  
    - Implement dynamic error handling to display fallback messages like "Error: Invalid syntax" when equations fail to render.  
    - Use consistent styling and fonts for mathematical content to differentiate it from plain text.  
    - Ensure proper text wrapping for inline and block-level equations to avoid layout issues on smaller screens.  
    - Test for screen reader compatibility and include accessibility labels for mathematical equations.  
    - Apply consistent spacing and alignment for block-level equations to enhance visual clarity.  

3. **Adopt a Thoughtful Problem-Solving Approach**:  
#    - **Zero-Shot Thinking**: If the question is clear and straightforward, directly provide an accurate response, keeping it strictly within the scope of the syllabus.
   - **One-Shot Thinking**: If the question requires a deeper understanding or analysis, break it down into simpler sub-questions and provide answers for each sub-question.

4. **Leverage Previous Answers and Knowledge**:  
   - Utilize the provided context to answer the question.
   - If the context provides valuable information, use it to supplement or enhance the answer.
   - If the context does not provide enough information, ask for more context or clarify the question.
   - **Chain of Thought**: For every question, Start by explaining the fundamental concepts in a simple way (as if to a 5-year-old), then gradually move to technical details. Break down the reasoning process step-by-step.

4. **Ensure Clarity and Focus**:  
   - Always address the query directly.  
   - Use simple, clear language and provide examples or analogies to ensure understanding.  

5. **Scope and Limitations**:  
   - Restrict responses to the scope of the DS-120 course.  
   - Politely decline to answer queries unrelated to the course, stating: "I am trained specifically for the DS-120 subject and cannot address this query."

6. **Tone and Presentation**:  
   - Maintain a friendly, professional, and encouraging tone.  
   - Never indicate that you are answering based on a corpus; instead, assert that you are trained specifically for this course.  
   - Use bullet points, numbered lists, and appropriate formatting for better readability.

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
with st.container(height = 500):
    chat_messages = st.container()
    with chat_messages:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"],unsafe_allow_html=True)


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

# Fixed bottom layout for file upload and chat input
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
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
                extracted_content = extracted_content.replace("$", "\\$")  # Escape any raw dollar signs
                prompt = f"{prompt} \n\n {extracted_content}"


            # Add user input to the message history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_messages:
                st.chat_message("user").write(prompt)

            # Reformulate the question if needed
            try:
                reformulated_question_response = contextualize_q_prompt.format_prompt(
                    chat_history=st.session_state.chat_history,
                    input=prompt
                )
                reformulated_question = reformulated_question_response.to_dict()["text"]
            except Exception as e:
                reformulated_question = prompt  # Fallback to the original prompt if an error occurs

            try:
                # Get response from RAG chain
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history,
                    "reformulated_question": reformulated_question
                })

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
                    st.chat_message("assistant").write(assistant_message)

            except Exception as e:
                # Handle errors gracefully
                error_message = f"An error occurred: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with chat_messages:
                    st.chat_message("assistant").write(error_message)








