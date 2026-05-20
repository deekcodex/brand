# ==========================================
# GLOBAL PRODUCT MARKETING CHATBOT
# Advanced Streamlit App
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

# ==========================================
# TITLE
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
# PRODUCT INPUT
# ==========================================

product_name = st.text_input(
    "Enter Product Name",
    placeholder="Example: iPhone 16 Pro"
)

# ==========================================
# QUESTION TYPE
# ==========================================

question_type = st.selectbox(
    "Select Requirement",
    [
        "Product Details",
        "Marketing Slogan",
        "Target Audience",
        "Product Features",
        "Advertisement",
        "Instagram Caption",
        "SEO Keywords"
    ]
)

# ==========================================
# GENERATE FUNCTION
# ==========================================

def generate_response(product, question):

    prompt = f"""
    Product Name: {product}

    Market: {country}

    Marketing Tone: {tone}

    Task: {question}

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

if st.button("Generate AI Marketing Content"):

    if product_name == "":
        st.warning("Please enter product name")

    else:

        with st.spinner("Generating Global Marketing Content..."):

            answer = generate_response(
                product_name,
                question_type
            )

        # ======================================
        # OUTPUT
        # ======================================

        st.success("Content Generated Successfully")

        st.subheader("🤖 AI Marketing Response")

        st.write(answer)

        # ======================================
        # EXTRA GENERATED CONTENT
        # ======================================

        st.markdown("---")

        # Slogan
        slogan_prompt = f"""
        Create a powerful global marketing slogan
        for {product_name}.
        """

        slogan = generator(
            slogan_prompt,
            max_length=50
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
            max_length=100
        )[0]["generated_text"]

        st.subheader("📱 Social Media Caption")

        st.success(caption)

        # SEO Keywords
        seo_prompt = f"""
        Generate SEO keywords for {product_name}.
        """

        seo = generator(
            seo_prompt,
            max_length=80
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
