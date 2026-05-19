# ============================================
# International Business Marketing Prompt App
# WORKING STREAMLIT CODE
# ============================================

import streamlit as st
import sys
import os

# --- BEGIN WORKAROUND FOR MODULE NOT FOUND ERROR IN COLAB SUBPROCESSES ---
# Add the site-packages directory to sys.path to ensure packages are found.
# This is a workaround for potential environment issues with subprocess.Popen and streamlit in Colab.
# sys.path should already contain it, but sometimes explicit addition is needed.
if '/usr/local/lib/python3.12/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
# --- END WORKAROUND ---

from transformers import pipeline

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Marketing Prompt App",
    page_icon="🌍",
    layout="centered"
)

# ============================================
# TITLE
# ============================================

st.title("🌍 International Business Marketing Prompt Application")

st.write(
    "Generate Product Titles, Marketing Slogans, "
    "and Advertising Content using AI."
)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    text_generator = pipeline(
        "text-generation", # Changed from "text2text-generation" to "text-generation"
        model="google/flan-t5-small"
    )

    return text_generator

generator = load_model()

# ============================================
# USER INPUT
# ============================================

product_name = st.text_input(
    "Enter Product Name",
    "GlowFresh Herbal Face Wash"
)

target_market = st.selectbox(
    "Select Target Market",
    [
        "Global",
        "India",
        "USA",
        "Europe",
        "Asia"
    ]
)

# ============================================
# GENERATE FUNCTION
# ============================================

def generate_text(prompt):

    result = generator(
        prompt,
        max_length=100
    )

    return result[0]["generated_text"]

# ============================================
# BUTTON
# ============================================

if st.button("Generate Content"):

    # ----------------------------------------
    # Product Title
    # ----------------------------------------

    title_prompt = f"""
    Create a professional global product title
    for {product_name}.
    """

    title = generate_text(title_prompt)

    # ----------------------------------------
    # Marketing Slogan
    # ----------------------------------------

    slogan_prompt = f"""
    Create a catchy marketing slogan
    for {product_name}.
    """

    slogan = generate_text(slogan_prompt)

    # ----------------------------------------
    # Branding Expert
    # ----------------------------------------

    branding_prompt = f"""
    Act as a branding expert.

    Write a professional advertisement
    for {product_name} targeting {target_market}.
    """

    branding = generate_text(branding_prompt)

    # ----------------------------------------
    # Digital Marketing Expert
    # ----------------------------------------

    digital_prompt = f"""
    Act as a digital marketing expert.

    Create a social media advertisement
    for {product_name}.
    """

    digital = generate_text(digital_prompt)

    # ----------------------------------------
    # Customer Psychology Expert
    # ----------------------------------------

    psychology_prompt = f"""
    Act as a customer psychology expert.

    Write an emotional advertisement
    for {product_name}.
    """

    psychology = generate_text(psychology_prompt)

    # ========================================
    # OUTPUTS
    # ========================================

    st.success("Content Generated Successfully")

    st.subheader("🌟 Global Product Title")
    st.info(title)

    st.subheader("📢 Marketing Slogan")
    st.success(slogan)

    st.subheader("📝 Expert Perspectives")

    tab1, tab2, tab3 = st.tabs([
        "Branding Expert",
        "Digital Expert",
        "Psychology Expert"
    ])

    with tab1:
        st.write(branding)

    with tab2:
        st.write(digital)

    with tab3:
        st.write(psychology)

    # ========================================
    # SOCIAL MEDIA CAPTION
    # ========================================

    st.subheader("📱 Social Media Caption")

    caption = f"""
✨ Try {product_name} today!

Feel fresh, clean, and confident every day.

#FaceWash #Skincare #GlowFresh #Beauty
"""

    st.code(caption)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.write("Developed using Streamlit + Hugging Face")
