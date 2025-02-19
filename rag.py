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
# os.environ["LANGSMITH_TRACING_V2"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "Tutor"

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
def invoke_chain(user_prompt: str, chat_history: list):
   
    formatted_history = format_chat_history(chat_history)
    context_prompt = [
                      f"Chat History:\n{formatted_history}\n\n"
                      f"User Query: {user_prompt}\n\n"
                      "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. "
                      "Do NOT answer the question, just reformulate it if needed or otherwise return it as is."
                      
    ]
    print(context_prompt)
    new_question = chat_llm(context_prompt).content
    print(new_question)
    retrieved_docs = retriever.invoke(new_question)
    raw_docs = format_retrieved_docs(retrieved_docs)
    scope_prompt = [
        f"""Given the user query and the retrieved documents from the course material for the user query, determine the category of the question. Classify the query to one of the given categories: 
           - 'out of scope': The query is Out of Scope of the Course Material.
           - 'question': A specific problem requiring a numerical solution.
           - 'concept': An explanation of a mathematical or data science concept.
           - 'miscellaneous': Queries that do not fit the first two categories.
        Only return One of 'out of scope','question', 'concept', or 'miscellaneous'.
        User Query: {new_question}
        Retrieved Documents: {raw_docs}
        """
    ]    

    result = chat_llm(scope_prompt).content
    category = result.replace("'", "").strip().lower()
    print("Category: " + category)

    if category == "question":
        # Extract key concepts from the user query using your specified approach.
        extractor_result = concept_extractor(question=new_question)
        extracted_concepts = [c.strip() for c in extractor_result.concepts.split(",") if c.strip()]
        print(f"Extracted Concepts: {extracted_concepts}")  # Debugging

        # Retrieve documents for each extracted concept.
        combined_chunks = []
        for concept in extracted_concepts:
            concept_docs = retriever.invoke(concept, search_kwargs={"k": 4})
            formatted_docs = format_retrieved_docs(concept_docs)
            if formatted_docs.strip():
                combined_chunks.append(formatted_docs)
        raw_docs_question = "\n\n".join(combined_chunks) if combined_chunks else "No relevant documents found."
        
        # Assignment logic check.
        # if "Assignment" in raw_docs_question:
        #     tool_used = "AssignmentGuidance"
        #     guidance_prompt = (
        #         f"User Query: {user_prompt}\n\n"
        #         "Instructions: You are a tutor who guides students in problem-solving without providing direct answers. "
        #         "Provide a structured, step-by-step explanation of how the student should approach solving this problem. "
        #         "Do not give the final numerical solution; instead, explain the concepts, formulas, and steps they need to use. "
        #         "At the end, mention that assignment problems cannot be solved completely."
        #     )
        #     response = chat_llm(guidance_prompt).content
        # else:
            # Determine the method for solving the query using a normal LLM call.
        method_prompt = (
                f"Extracted Class Material for Key Concepts:\n{raw_docs_question}\n\n"
                f"User Query: {new_question}\n\n"
                "Instructions: Based solely on the above class material, identify the key method or approach taught in the course that applies to solving the query. "
                "Your response should be based on detailed explanations in the material (not merely inferred from headings or titles) and should include any relevant formulas, steps, or reasoning. "
                "Output only a concise and clear description of the method."
        )
        method_response = chat_llm(method_prompt).content
        solving_method = method_response.strip()

            
            # Build the final prompt including the extracted concepts, the suggested method, and the aggregated context.
        prompt_text = (
            f"Suggested Method: {solving_method}\n\n"
            f"User Query: {user_prompt}\n\n"
                "Instructions: You are a Mathematics tutor for a College Subject DS-120. "
                "Provide a detailed, step-by-step solution using the suggested method. "
                "Do not include any external knowledge, and include the extracted concepts for reference."
            )
        response = chat_llm(prompt_text).content
        print(f"\n\nRetrieved Documents:\n" + raw_docs_question )
        print("================================")
        print(f"\n\nsuggested method:\n" + solving_method )
        
        
    elif category == "concept":
        prompt_text = (
            f"Retrieved Class Material:\n{raw_docs}\n\n"
            f"User Query: {new_question}\n\n"
            "Instructions: You are a tutor meant to explain concepts for this Fundamentals of Data Science course. Imagine that you are explaining the query to a 5 year old. Provide an intuitive, step-by-step explanation of the concept and gradually transition to mathematical explainations, using only the class material above.  "
            "Keep your explanation simple, clear, and accessible as if explaining to a young learner. "
            "Do not assume anything and do not include any external knowledge No coding related information is to be given"
        )
        print("================================")
        print(prompt_text)
        print("================================")
        response = chat_llm(prompt_text).content

    elif category == "miscellaneous":
        prompt_text = (
            f"Query: {new_question}\n\n"
            "Instructions: You are a tutor trained to answer course-related queries of the subject DS-120 only. "
            "If the query is a greeting or trivial message,respond appropriately in a polite and friendly tone. "
            "If the query is anything else politely state that you only handle course-related questions."
        )
        print("================================")
        print(prompt_text)
        print("================================")
        response = chat_llm(prompt_text).content
    
    else:
        response = "This query seems out of scope, try asking something that has been taught in the class"

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






