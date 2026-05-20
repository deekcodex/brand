# ==========================================
# GLOBAL PRODUCT MARKETING CHATBOT
# Advanced Streamlit App with UI/UX Enhancements
# ==========================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Global Product Marketing Chatbot",
    page_icon="🌍",
    layout="wide"
)

# Add Bootstrap CSS
st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0486qhLnuZ2cdeRhO02iuK6FUUVM" crossorigin="anonymous">', unsafe_allow_html=True)

# ==========================================
# TITLE AND DESCRIPTION
# ==========================================

st.title("🌍 Global Product Marketing Chatbot")

st.markdown("""
This AI chatbot generates:

✅ Product Details
✅ Product Features
✅ Global Marketing Slogans
✅ Target Audience
✅ International Advertisement
✅ Social Media Captions
✅ SEO Keywords
""")

with st.expander("💡 How it works"): # UI/UX: Add an expander for instructions
    st.write("""
    Simply enter your product name, select your target market and desired marketing tone. 
    Then, ask a question about the product, and the AI will generate a detailed response!
    You'll also get bonus marketing slogans, social media captions, and SEO keywords.
    """)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    class CustomGenerator:
        def __init__(self, tokenizer, model):
            self.tokenizer = tokenizer
            self.model = model

        def __call__(self, prompt, max_length, do_sample, temperature):
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(**inputs, max_length=max_length, do_sample=do_sample, temperature=temperature)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return [{"generated_text": generated_text}]

    generator = CustomGenerator(tokenizer, model)
    return generator

generator = load_model()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("🌐 Marketing Settings")

country = st.sidebar.selectbox(
    "Select Market",
    [
        "Global",
        "India",
        "USA",
        "Europe",
        "Asia",
        "Middle East"
    ]
)

tone = st.sidebar.selectbox(
    "Marketing Tone",
    [
        "Professional",
        "Luxury",
        "Modern",
        "Emotional",
        "Friendly"
    ]
)

# ==========================================
# MAIN CONTENT AREA (UI/UX: Using columns for better layout)
# ==========================================

col1, col2 = st.columns(2) # Create two columns

with col1:
    product_name = st.text_input(
        "Enter Product Name",
        placeholder="Example: iPhone 16 Pro"
    )

with col2:
    # Changed from selectbox to text_area for conversational input
    user_query = st.text_area(
        "Ask me about the product (e.g., 'What are its key features?', 'Who is the target audience?')",
        placeholder="Example: 'What are the main benefits of this product?'"
    )

# ==========================================
# GENERATE FUNCTION
# ==========================================

def generate_response(product, query):

    prompt = f"""
    Product Name: {product}

    Market: {country}

    Marketing Tone: {tone}

    Task: {query}

    Give a detailed and professional response.
    """

    result = generator(
        prompt,
        max_length=250,
        do_sample=True,
        temperature=0.8
    )

    return result[0]["generated_text"]

# ==========================================
# BUTTON
# ==========================================

if st.button("Generate AI Marketing Content", use_container_width=True): # UI/UX: Button spans full width

    if product_name == "":
        st.warning("Please enter product name")
    elif user_query == "": # Ensure user provides a query
        st.warning("Please enter a question about the product.")

    else:

        with st.spinner("Generating Global Marketing Content..."):

            answer = generate_response(
                product_name,
                user_query # Use the user's free-form query
            )

        # ======================================
        # OUTPUT (UI/UX: Organize output with sections)
        # ======================================

        st.success("Content Generated Successfully")

        st.subheader("🤖 AI Marketing Response")

        st.write(answer)

        st.markdown("---")
        st.subheader("✨ Bonus Marketing Content") # UI/UX: Clearer separation

        # Slogan
        slogan_prompt = f"""
        Create a powerful global marketing slogan
        for {product_name}.
        """

        slogan = generator(
            slogan_prompt,
            max_length=50,
            do_sample=True,
            temperature=0.8
        )[0]["generated_text"]

        st.subheader("📢 Marketing Slogan")

        st.info(slogan)

        # Social Caption
        caption_prompt = f"""
        Create Instagram marketing caption
        for {product_name}.
        """

        caption = generator(
            caption_prompt,
            max_length=100,
            do_sample=True,
            temperature=0.8
        )[0]["generated_text"]

        st.subheader("📱 Social Media Caption")

        st.success(caption)

        # SEO Keywords
        seo_prompt = f"""
        Generate SEO keywords for {product_name}.
        """

        seo = generator(
            seo_prompt,
            max_length=80,
            do_sample=True,
            temperature=0.8
        )[0]["generated_text"]

        st.subheader("🔍 SEO Keywords")

        st.code(seo)

# ==========================================
# SAMPLE PRODUCTS
# ==========================================

st.markdown("---")

st.subheader("🌟 Example Products")

products = [
    "iPhone 16 Pro",
    "Samsung Galaxy S25",
    "Nike Running Shoes",
    "GlowFresh Face Wash",
    "Tesla Model Y",
    "Sony Headphones"
]

for item in products:
    st.write("✅", item)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.write("Developed using Streamlit + Hugging Face")
