import streamlit as st
from dotenv import load_dotenv
from rag import invoke_rag_chain,update_chat_history
from elements import add_instructions_button , process_uploaded_file , sanitize_latex

# Load prompts from secrets.toml
contextualize_q_system_prompt = st.secrets["ds120_prompts"]["contextualize_q_system_prompt"]
qa_prompt_template = st.secrets["ds120_prompts"]["qa_prompt_template"]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize the message history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # Initialize the chat history
if "chat_input" not in st.session_state:
    st.session_state.chat_input = False


# Streamlit app setup
st.set_page_config(
    page_title="DS-120 Virtual Teaching Assistant Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("<h1 style='text-align: center;'>DS-120 Virtual Teaching Assistant Chatbot</h1>", unsafe_allow_html=True)

# Hide Deploy button and three-dot menu but keep "Running"
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;} /* Hide the three-dot menu */
        footer {visibility: hidden;} /* Hide Streamlit footer */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add an "Instructions" button to the sidebar
st.markdown(
    """
    <style>
        /* Reduce the sidebar width when expanded */
        [data-testid="stSidebarContent"][aria-expanded="true"] > div:first-child {
            width: 100px;  /* Set a small width for the sidebar */
            padding: 0;  /* Remove internal padding */
        }

        /* Hide the sidebar completely when collapsed */
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 0px;  /* Collapse width */
            margin-left: -150px;  /* Adjust for alignment */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def disable_callback():
    st.session_state.chat_input = True

# Call the function to display the button in the sidebar
add_instructions_button()


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
    # with col1:
    #     st.markdown('<div class="stFileUploader">', unsafe_allow_html=True)
    #     uploaded_file = st.file_uploader(label_visibility="collapsed", label="Upload a file (image or PDF)", key="file_uploader",type = ['png','pdf'])

    #     extracted_content = ""
    #     if uploaded_file:
    #         extracted_content = process_uploaded_file(uploaded_file)
    
    with col2:
            st.markdown('<div class="custom-col">', unsafe_allow_html=True)
            prompt = st.chat_input("Type your question here...", disabled=st.session_state.chat_input, on_submit=disable_callback)
            if prompt:
                st.session_state.chat_input = True  # Disable input while processing
                st.session_state["is_processing"] = True
                # Check if an image is uploaded and extract content
                # if extracted_content:
                #     ui_display_prompt = f"{prompt} \n\n[Attachment]"
                #     prompt = f"{prompt}\n\n{extracted_content}"
                # else:
                ui_display_prompt = prompt

                # Add user input (with "Attachment" if applicable) to the message history
                st.session_state.messages.append({"role": "user", "content": ui_display_prompt})

                # ✅ Render messages inside the chat container to maintain aesthetics
                with chat_messages:
                    st.chat_message("user").write(ui_display_prompt)

                    # ✅ Use `st.empty()` so response replaces "Thinking..." without breaking UI
                    response_placeholder = st.chat_message("assistant").empty()

                    with response_placeholder:
                        with st.spinner("Thinking..."):
                            try:
                                response = invoke_rag_chain(prompt, st.session_state.chat_history)
                                response_text = response["answer"]
                                response_text = sanitize_latex(response_text)
                                #response_text = add_newline_after_block_math(response_text)
                                print("\n===== RAW SANITIZED OUTPUT =====\n")
                                print(response_text)
                                print("\n===============================\n")
                            except Exception as e:
                                response_text = f"An error occurred: {str(e)}"

                    # ✅ Overwrite "Thinking..." with actual response (inside the chat UI)
                    response_placeholder.write(response_text)

                    # Add assistant response to session state
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Update chat history to maintain context
                st.session_state.chat_history = update_chat_history(
                    st.session_state.chat_history,
                    prompt,
                    response_text
                )

                # Reset processing state
                st.session_state.chat_input = False  # Enable input after processing
                st.rerun()


                                