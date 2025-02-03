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


load_dotenv()


os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Tutor"

# Load API keys from secrets.toml
# Correct key names based on secrets.toml
openai_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
langsmith_key = st.secrets["api_keys"]["LANGSMITH_API_KEY"]

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LANGSMITH_API_KEY"] = langsmith_key  #

@traceable
def invoke_rag_chain(prompt: str, chat_history: list):
    # Step 1: Run the retriever to get context (no LLM call yet)
    retrieved_context = history_aware_retriever.invoke({"input": prompt, "chat_history": chat_history})

    # Step 2: Check for restricted keywords in the context
    should_remove_context = False
    for doc in retrieved_context:
        if hasattr(doc, 'page_content'):
            if "assignment: do not answer" in doc.page_content.lower():
                should_remove_context = True
                break

    # Step 3: Set assignment instruction
    if should_remove_context:
        assignment_instruction = (
        "📚 *Assignment Notice:* This question is part of an assignment. "
        "You *MUST NOT* provide any direct answers, solutions, or calculations. "
        "Instead, you are expected to *explain the underlying concepts* related to the question in detail. "
        "If you feel unsure due to missing context, simply inform the user: "
        "*'I cannot provide direct answers to assignment questions, but I’m happy to explain the underlying concepts.'* "
        "*DO NOT* express confusion about missing context under any circumstances."
    )
        print("\n===== ASSIGNMENT DETECTED: CONTEXT REMOVED =====\n")
    else:
        assignment_instruction = "No special instructions."
        print("\n===== NO ASSIGNMENT DETECTED: CONTEXT INCLUDED =====\n")

    # Step 4: Prepare the final payload to send to the LLM
    if should_remove_context:
        # Remove context completely if assignment-related material is detected
        llm_payload = {
            "input": prompt,
            "context": "",  # Send empty context
            "chat_history": [],
            "assignment_instruction": assignment_instruction
        }
    else:
        # Send full context if no assignment is detected
        llm_payload = {
            "input": prompt,
            "context": retrieved_context,
            "chat_history": chat_history,
            "assignment_instruction": assignment_instruction
        }

    # Step 5: Send the prompt to the LLM
    try:
        response = question_answer_chain.invoke(llm_payload)

        # Debug: Check the structure of the LLM response
        print("\n===== LLM RESPONSE RECEIVED =====\n")
        print(f"Type of response: {type(response)}\n")
        print(f"Content of response: {response}\n")
        print("\n=================================\n")

        # Ensure consistent response format
        if isinstance(response, str):
            response = {"answer": response}
        elif isinstance(response, dict):
            if "answer" not in response:
                response = {"answer": response.get("output", "No answer found.")}

    except Exception as e:
        print(f"\n===== ERROR DURING LLM CALL: {e} =====\n")
        response = {"answer": f"An error occurred: {str(e)}"}

    return response


modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# Load FAISS vector store
vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":4})

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"], temperature=0.01)

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
        #MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Create a question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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