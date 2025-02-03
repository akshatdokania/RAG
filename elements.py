import streamlit as st
import re
import streamlit as st
from pix2text import Pix2Text
from pdf2image import convert_from_path
import PyPDF2
import tempfile




def add_instructions_button():
    # Instructions Button (Opens Info Box)
    if st.sidebar.button("Instructions", key="instructions_button"):
        st.sidebar.info(
            "### Instructions\n"
            "1. Upload files if needed.\n"
            "2. Type your question in the input box.\n"
            "3. Receive a response from the assistant."
        )

    # Feedback Button (Opens Link)
    if st.sidebar.button("Feedback", key="feedback_button"):
        st.sidebar.markdown(
            '[Click here to provide feedback](https://docs.google.com/forms/d/e/1FAIpQLSfaGG9AL_V0ThQ45mMO1bKDn_gLljhY1kAG1RY_k3E8U1Kefw/viewform)', 
            unsafe_allow_html=True
        )
        

def add_newline_after_block_math(text):
    """
    Ensures:
    - Block equations ($$ ... $$) are properly spaced:
      - A blank line after opening $$
      - A blank line before closing $$
      - A blank line after the entire block equation
    - Markdown headings (#, ##, ###, ####) are correctly separated from other content.
    - Prevents headings from merging with equations.
    """

    # ✅ Ensure **a blank line after opening $$ and before closing $$**
    text = re.sub(r"\$\$(.*?)\$\$", r"$$\n\1\n$$", text, flags=re.DOTALL)

    # ✅ Ensure **at least one blank line after block equations**
    text = re.sub(r"(\$\$.*?\$\$)(\S)", r"\1\n\n\2", text, flags=re.DOTALL)

    # ✅ Ensure **at least one blank line before Markdown headings**
    text = re.sub(r"(\$\$.*?\$\$)\s*(#+ )", r"\1\n\n\2", text, flags=re.DOTALL)

    return text


def sanitize_latex(text):
    """
    Step 9: Ensures proper LaTeX formatting for Streamlit Markdown.
    - Converts \( ... \) to $ ... $ (for inline math).
    - Converts \[ ... \] to $$ ... $$ (for block math).
    - Removes unnecessary spaces inside $$ ... $$.
    - Ensures block math appears on separate lines.
    - Prevents Streamlit rendering issues by enforcing correct newlines.
    """

    # 1️⃣ Skip processing if already wrapped in block math
    stripped_text = text.strip()
    if stripped_text.startswith("$$") and stripped_text.endswith("$$"):
        return text  # ✅ Don't modify already formatted block math

    # 2️⃣ Convert inline LaTeX `\( ... \)` → `$ ... $`
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)

    # 3️⃣ Convert block LaTeX `\[ ... \]` → `$$ ... $$`
    text = re.sub(r"\s*\\\[\s*(.*?)\s*\\\]\s*", r"\n$$\1$$\n", text, flags=re.DOTALL)


    # 5️⃣ Ensure inline math has proper spacing
    text = re.sub(r"(?<!\s)\$(.*?)\$(?!\s)", r" $\1$ ", text)  # ✅ Prevents missing spaces around math

    # 6️⃣ Ensure matching `$` delimiters (close any unclosed inline math)
    text = re.sub(r"\$\s+\$", "$$", text)
    if text.count("$") % 2 != 0:
        text += "$"  # ✅ Auto-close inline math if an odd `$` count is detected
    text = re.sub(r"\*\* \$(.*?)\$ \*\*", r"**$\1$**", text)
    #text = re.sub(r"\$\$(.*?)\$\$", r"$$\n\1\n$$", text, flags=re.DOTALL)
    def format_block_math(match):
        """Ensures block math expressions are formatted correctly."""
        content = match.group(1).strip()
        return f"\n$$\n{content}\n$$\n"
    text = re.sub(r"(\$[^$]+\$)\s*\n?\s*(\$\$)", r"\1\n\2", text)
    text = re.sub(r"\$\$(.*?)\$\$", format_block_math, text, flags=re.DOTALL)
    return text


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