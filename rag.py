import os
import streamlit as st
from dotenv import load_dotenv
import dspy
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

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Tutor"

openai_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_key

# DSPy Language Model Initialization (for Manager and Concept Extraction)
lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_key)
dspy.configure(lm=lm)

# LangChain ChatOpenAI Initialization (for question solving, concept explanation, and miscellaneous queries)
chat_llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_key)

# Load FAISS vector store
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=modelPath)
vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})


### Step 1: Define DSPy Agents for Manager and Concept Extraction

class ManagerSignature(dspy.Signature):
    """
    Classifies the user’s input into one of the following categories:
    - 'question': A specific problem requiring a numerical solution.
    - 'concept': An explanation of a mathematical or data science concept.
    - 'miscellaneous': Queries that do not fit the first two categories.
    """
    prompt = dspy.InputField(desc="User's input")
    category = dspy.OutputField(desc="One of 'question', 'concept', or 'miscellaneous'")

class ConceptExtractionSignature(dspy.Signature):
    """
    Extract the key concepts from the domain of data science required to solve the question,
    based only on the course material. Output a comma-separated list of concepts (e.g., "conditional probability, combinations").
    """
    question = dspy.InputField()
    concepts = dspy.OutputField(desc="Comma-separated list of key concepts")

# Instantiate DSPy Predictive Models for the Manager and Concept Extractor
manager_agent = dspy.Predict(ManagerSignature)
concept_extractor = dspy.Predict(ConceptExtractionSignature)


### Step 2: Format Retrieved Documents

def format_retrieved_docs(docs):
    """Formats retrieved FAISS documents into clean text."""
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."


def format_chat_history(chat_history):
    """Formats chat history for use in DSPy prompts."""
    history_text = "\n".join(
        [f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}" for msg in chat_history]
    )
    return history_text[-1500:]

### Step 3: Implement the Integrated Retrieval Chain
@traceable
def invoke_chain(user_prompt: str, chat_history: list):
    """
    Processes user input by first classifying it via the DSPy manager agent, then:
      - For questions, uses the DSPy concept extractor and a ChatOpenAI prompt.
      - For concept explanations and miscellaneous queries, uses ChatOpenAI with custom prompts.
    """
    # 1. Classify the user input using the DSPy Manager Agent.
    result = manager_agent(prompt=user_prompt)
    category = result.category.strip().lower()

    formatted_history = format_chat_history(chat_history)
    # 2. Retrieve documents from the FAISS index.
    retrieved_docs = retriever.invoke(user_prompt)
    raw_docs = format_retrieved_docs(retrieved_docs)

    tool_used = "None"
    response = "I don't have enough information to answer this."

    if category == "question":
        
        # Step 1: Extract key concepts from the user query.
        extractor_result = concept_extractor(question=user_prompt)
        extracted_concepts = [c.strip() for c in extractor_result.concepts.split(",") if c.strip()]
        print(f"Extracted Concepts: {extracted_concepts}")  # Debugging
        
        # Step 2: Retrieve documents for each extracted concept.
        combined_chunks = []
        raw_docs_question=[]
        for concept in extracted_concepts:
            concept_docs = retriever.invoke(concept, search_kwargs={"k": 3})
            formatted_docs = format_retrieved_docs(concept_docs)
            if formatted_docs.strip():
                combined_chunks.append(formatted_docs)
        
        # Step 3: Combine all retrieved documents into a single aggregated context.
        raw_docs_question = "\n\n".join(combined_chunks) if combined_chunks else "No relevant documents found."
        
        # Step 4: Assignment logic check (after concept extraction).
        if "Assignment" in raw_docs_question:
            tool_used = "AssignmentGuidance"
            guidance_prompt = (
                f"User Query: {user_prompt}\n\n"
                f"Chat History:\n{formatted_history}\n\n"
                "Instructions: You are a tutor who guides students in problem-solving without providing direct answers. "
                "Provide a structured, step-by-step explanation of how the student should approach solving this problem. "
                "Do not give the final numerical solution, but explain the concepts, formulas, and steps they need to use. "
                "At the end, also mention that assignment problems cannot be solved completely."
            )
            print("================================")
            print(f"Context{raw_docs_question}")
            print("================================")
            print(guidance_prompt)
            print("================================")
            response = chat_llm(guidance_prompt).content
        else:
            # Step 5: Build the prompt using the extracted concepts and retrieved context.
            prompt_text = (
                f"Extracted Concepts: {', '.join(extracted_concepts)}\n\n"
                f"Chat History:\n{formatted_history}\n\n"
                f"Retrieved Context:\n{raw_docs_question}\n\n"
                f"User Query: {user_prompt}\n\n"
                "Instructions: You are a teaching assistant for Fundamentals of Data Science course and will be using a REACT approach. Use the retrieved class material above to answer the query. "
                "If the information is sufficient, provide a detailed and complete solution using the methods defined in the class material. "
                "If the information is insufficient, explain why and state that the topic is outside the scope of this tutor. "
                "Do not include any external knowledge, and include the extracted concepts for reference. No coding related information is to be given"
            )
            print("================================")
            print(prompt_text)
            print("================================")
            response = chat_llm(prompt_text).content
    
    
    
   

    elif category == "concept":
        prompt_text = (
            f"Retrieved Class Material:\n{raw_docs}\n\n"
            f"Chat History:\n{formatted_history}\n\n"
            f"User Query: {user_prompt}\n\n"
            "Instructions: You are a tutor meant to explain concepts for this Fundamentals of Data Science course. Imagine that you are explaining the query to a 5 year old. Provide an intuitive, step-by-step explanation of the concept and gradually transition to mathematical explainations, using only the class material above.  "
            "Keep your explanation simple, clear, and accessible as if explaining to a young learner. "
            "If the information is insufficient, explain why and state that the topic is outside the scope of this tutor. "
            "Do not include any external knowledge, and include the extracted concepts for reference. No coding related information is to be given"
        )
        print("================================")
        print(prompt_text)
        print("================================")
        response = chat_llm(prompt_text).content

    elif category == "miscellaneous":
        prompt_text = (
            f"Chat History:\n{formatted_history}\n\n"
            f"Query: {user_prompt}\n\n"
            "Instructions: You are a tutor trained to answer course-related queries of the subject DS-120 only. "
            "If the query is a greeting or trivial message,respond appropriately in a polite and friendly tone. "
            "If the query is anything else politely state that you only handle course-related questions."
        )
        print("================================")
        print(prompt_text)
        print("================================")
        response = chat_llm(prompt_text).content

    return response


### Step 4: Update Chat History Function

def update_chat_history(chat_history, human_message, ai_message):
    chat_history.extend(
        [
            HumanMessage(content=human_message),
            AIMessage(content=ai_message),
        ]
    )
    if len(chat_history) > 5:
        chat_history = chat_history[-5:]
    return chat_history
