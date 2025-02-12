import os
import streamlit as st
from dotenv import load_dotenv
import dspy
import re
import numpy as np
import sympy as sp
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Tutor"

openai_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_key

# DSPy Language Model Initialization
lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_key)
dspy.configure(lm=lm)

# Load FAISS vector store
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=modelPath)
vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})


### Step 1: Define DSPy Agents

class ManagerSignature(dspy.Signature):
    """Classifies user input into 'question', 'concept', or 'miscellaneous'."""
    prompt = dspy.InputField(desc="User's input")
    category = dspy.OutputField(desc="One of 'question', 'concept', or 'miscellaneous'")

class QuestionSolverSignature(dspy.Signature):
    """Solves math problems step-by-step."""
    problem = dspy.InputField()
    solution = dspy.OutputField(desc="Step-by-step solution.")

class ConceptExplainerSignature(dspy.Signature):
    """Explains concepts using chain-of-thought reasoning."""
    concept = dspy.InputField()
    explanation = dspy.OutputField(desc="Structured concept explanation.")

class MiscellaneousSignature(dspy.Signature):
    """Handles general queries using FAISS knowledge."""
    query = dspy.InputField()
    response = dspy.OutputField(desc="Relevant response based on retrieved knowledge.")

class AssignmentGuidanceSignature(dspy.Signature):
    """Guides users on how to approach assignment-related problems."""
    problem = dspy.InputField()
    guidance = dspy.OutputField(desc="Step-by-step explanation without solving.")

class ConceptExtractionSignature(dspy.Signature):
    """
    Extract the key concepts from the domain of data science required to solve the question, based only on the course material.
    Output a comma-separated list of concepts (e.g., "conditional probability, combinations").
    """
    question = dspy.InputField()
    concepts = dspy.OutputField(desc="Comma-separated list of key concepts")

# Instantiate DSPy Predictive Models
manager_agent = dspy.Predict(ManagerSignature)
question_agent = dspy.Predict(QuestionSolverSignature)
concept_explainer = dspy.Predict(ConceptExplainerSignature)
misc_agent = dspy.Predict(MiscellaneousSignature)
assignment_guide = dspy.Predict(AssignmentGuidanceSignature)
concept_extractor = dspy.Predict(ConceptExtractionSignature)


### Step 2: Format Retrieved Documents
def format_retrieved_docs(docs):
    """Formats retrieved FAISS documents into clean text."""
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."


### Step 3: Implement DSPy-Based Retrieval Chain

def invoke_dspy_chain(user_prompt: str, chat_history: list):
    """Processes user input using DSPy instead of LangChain."""
    
    # Step 1: Classify user input
    category = manager_agent(prompt=user_prompt).category.strip().lower()
    
    # Step 2: Retrieve documents from FAISS
    retrieved_docs = retriever.invoke(user_prompt)
    raw_docs = format_retrieved_docs(retrieved_docs)

    tool_used = "None"
    response = "I don't have enough information to answer this."

    # Step 3: Process based on category

    ## (A) Question Handling
    if category == "question":
        tool_used = "QuestionSolver"

        # Assignment Handling
        if "Assignment" in raw_docs:
            tool_used = "AssignmentGuidance"
            guidance_prompt = (
                f"User Query: {user_prompt}\n\n"
                "Instructions: You are a tutor who guides students in problem-solving without providing direct answers. "
                "Provide a structured, step-by-step explanation of how the student should approach solving this problem. "
                "Do not give the final numerical solution, but explain the concepts, formulas, and steps they need to use. "
                "At the end, also mention that assignment problems cannot be solved completely."
            )
            guidance_result = assignment_guide(problem=guidance_prompt)
            response = getattr(guidance_result, "guidance", "I couldn't generate guidance on how to approach this problem.")
        else:
            # Extract key concepts from user query
            extractor_result = concept_extractor(question=user_prompt)
            extracted_concepts = extractor_result.concepts.split(", ")

            # Retrieve documents for extracted concepts
            combined_chunks = []
            for concept in extracted_concepts:
                concept_docs = retriever.invoke(concept)
                formatted_docs = format_retrieved_docs(concept_docs)
                if formatted_docs.strip():
                    combined_chunks.append(formatted_docs)

            # Combine all retrieved documents
            raw_docs = "\n\n".join(combined_chunks) if combined_chunks else "No relevant documents found."

            # Build question prompt
            prompt_text = (
                f"Extracted Concepts: {', '.join(extracted_concepts)}\n\n"
                f"Retrieved Context:\n{raw_docs}\n\n"
                f"User Query: {user_prompt}\n\n"
                "Instructions: Use the retrieved class material to solve the query. "
                "If the information is insufficient, explain why and state that the topic is outside the scope of this tutor."
            )

            # Get answer from question agent
            result = question_agent(problem=prompt_text)
            response = result.solution

    ## (B) Concept Explanation
    elif category == "concept":
        tool_used = "ConceptExplainer"
        prompt_text = (
            f"Retrieved Class Material:\n{raw_docs}\n\n"
            f"User Query: {user_prompt}\n\n"
            "Instructions: Provide an intuitive, step-by-step explanation of the concept using only the class material above. "
            "If the material is insufficient, state that the topic is outside the course scope."
        )
        result = concept_explainer(concept=prompt_text)
        response = result.explanation

    ## (C) Miscellaneous Queries
    elif category == "miscellaneous":
        tool_used = "MiscellaneousAgent"
        prompt_text = (
            f"Query: {user_prompt}\n\n"
            f"Retrieved Documents:\n{raw_docs}\n\n"
            "Instructions: Respond based on retrieved knowledge. "
            "If unrelated, politely state that you only handle course-related questions."
        )
        result = misc_agent(query=prompt_text)
        response = result.response

    return response



### Step 4: Update Chat History Function

def update_chat_history(chat_history, human_message, ai_message):
    chat_history.extend(
        [
            {"role": "user", "content": human_message},
            {"role": "assistant", "content": ai_message},
        ]
    )
    return chat_history[-5:]  # Keep last 5 messages

