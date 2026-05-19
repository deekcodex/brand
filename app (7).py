import streamlit as st
import sys
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- BEGIN WORKAROUND FOR MODULE NOT FOUND ERROR IN COLAB SUBPROCESSES ---
# Add the site-packages directory to sys.path to ensure packages are found.
# This is a workaround for potential environment issues with subprocess.Popen and streamlit in Colab.
site_packages_path = '/usr/local/lib/python3.12/dist-packages'
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)
# --- END WORKAROUND ---

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Product Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("🤖 AI Product Marketing Chatbot")

st.write(
    "Ask anything about your product like slogan, "
    "marketing ideas, advertisement, features, "
    "target audience, and social media captions."
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# ==========================================
# PRODUCT INPUT
# ==========================================

product_name = st.text_input(
    "Enter Product Name",
    placeholder="Example: GlowFresh Face Wash"
)

# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_area(
    "Ask Your Question",
    placeholder="Example: Give marketing slogan"
)

# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(product, user_question):

    prompt = f"""
    Product: {product}

    Question: {user_question}

    Give a professional and creative marketing answer.
    """

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==========================================
# BUTTON
# ==========================================

if st.button("Generate Answer"):

    if product_name == "":
        st.warning("Please enter product name")

    elif question == "":
        st.warning("Please enter your question")

    else:

        with st.spinner("Generating AI Response..."):

            answer = generate_answer(
                product_name,
                question
            )

        st.success("Response Generated Successfully")

        st.subheader("🤖 AI Response")

        st.write(answer)

# ==========================================
# EXAMPLES
# ==========================================

st.markdown("---")

st.subheader("📌 Example Questions")

examples = [
    "Give marketing slogan",
    "Create Instagram caption",
    "Who is target audience?",
    "Give advertisement content",
    "Generate SEO keywords",
    "Give product benefits",
    "Create social media post"
]

for item in examples:
    st.write("✅", item)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.write("Developed using Streamlit + Hugging Face")
