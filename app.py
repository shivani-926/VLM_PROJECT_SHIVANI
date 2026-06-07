import streamlit as st
import sys
import os
import json
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from src.preprocess import preprocess_image
from src.model import VLMModel

# ── Page Configuration ──
st.set_page_config(
    page_title="VLM Image Understanding",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Dark Mode ──
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #8b5cf6; /* Sleek purple to match your theme */
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #a6a6a6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: rgba(139, 92, 246, 0.1);
        border-left: 4px solid #8b5cf6;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .answer-box {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .info-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #a6a6a6;
        padding: 0.8rem 1.2rem;
        border-radius: 0.5rem;
        font-size: 0.95rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown('<div class="main-title">Vision-Language Model</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Image Understanding using BLIP — Captioning & Visual Question Answering</div>', unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🌌 Vision-Language Engine")
    st.divider()
    st.markdown("### 📋 Project Info")
    st.markdown("""
    - **Model:** BLIP (Salesforce)
    - **Tasks:** Captioning + VQA
    - **Framework:** PyTorch
    - **Hub:** Hugging Face
    """)
    st.divider()
    st.markdown("### ⚙️ Model Info")
    st.info("Models are cached locally.\nFirst run downloads ~2GB total.")

# ── Load Model (cached so it loads only once) ──
@st.cache_resource
def load_model():
    return VLMModel(config_path="config.yaml")

# ── Main Area: Side-by-Side Features ──
col_feat1, col_feat2 = st.columns(2)

with col_feat1:
    st.info("📸 **Image Captioning**\n\nAutomatically describe any image in natural language.")
    
with col_feat2:
    st.success("❓ **Visual Question Answering**\n\nAsk any question about an image and get an AI answer.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>👇 Upload an image to get started</h4>", unsafe_allow_html=True)

# ── Centered Uploader ──
col_up1, col_up2, col_up3 = st.columns([1, 2, 1])
with col_up2:
    uploaded_file = st.file_uploader(
        "Choose a JPG/PNG image",
        type=["jpg", "jpeg", "png"],
        help="Upload any clear image to analyze",
        label_visibility="collapsed"
    )

st.divider()

if uploaded_file is not None:
    # ── Load and display image ──
    image = Image.open(uploaded_file).convert("RGB")

    # Save uploaded image temporarily
    temp_path = "data/raw/streamlit_temp.jpg"
    os.makedirs("data/raw", exist_ok=True)
    image.save(temp_path)

    # ── Two Tabs ──
    tab1, tab2 = st.tabs(["📸 Image Captioning", "❓ Visual Question Answering"])

    # ════════════════════════════════════
    #   TAB 1: IMAGE CAPTIONING
    # ════════════════════════════════════
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🖼️ Your Image")
            st.image(image, use_container_width=True)
            st.caption(f"Filename: {uploaded_file.name} | Size: {image.size[0]}x{image.size[1]} px")

        with col2:
            st.markdown("#### 💬 Generated Caption")
            st.markdown("BLIP analyzes the entire image and generates a natural language description automatically.")

            if st.button("🚀 Generate Caption", key="caption_btn", use_container_width=True):
                with st.spinner("🤖 BLIP is analyzing your image..."):
                    try:
                        model = load_model()
                        pil_image = preprocess_image(temp_path)
                        caption = model.generate_caption(pil_image)

                        st.markdown(
                            f'<div class="result-box">📝 <strong>Caption:</strong><br><br>"{caption}"</div>',
                            unsafe_allow_html=True
                        )

                        # Save result
                        result = {
                            "task": "captioning",
                            "image": uploaded_file.name,
                            "caption": caption
                        }
                        os.makedirs("outputs", exist_ok=True)
                        with open("outputs/streamlit_results.json", "w") as f:
                            json.dump(result, f, indent=2)

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

            with st.expander("🧠 How it works behind the scenes"):
                st.markdown("The image is encoded by a Vision Transformer (ViT), then a language decoder generates the caption word by word using cross-attention.")

    # ════════════════════════════════════
    #   TAB 2: VISUAL QUESTION ANSWERING
    # ════════════════════════════════════
    with tab2:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🖼️ Your Image")
            st.image(image, use_container_width=True)
            st.caption(f"Filename: {uploaded_file.name} | Size: {image.size[0]}x{image.size[1]} px")

        with col2:
            st.markdown("#### ❓ Ask a Question")
            st.markdown("Type any question about the image below:")

            # Suggested questions
            st.markdown("**Quick questions to try:**")
            suggested = [
                "What is in the image?",
                "What color is the main object?",
                "How many objects are in the image?",
                "Is this indoors or outdoors?"
            ]

            # Clickable suggestion buttons
            cols = st.columns(2)
            selected_question = ""
            for i, q in enumerate(suggested):
                if cols[i % 2].button(q, key=f"q_{i}", use_container_width=True):
                    selected_question = q

            st.divider()

            # Manual question input
            question = st.text_input(
                "Or type your own question:",
                value=selected_question,
                placeholder="e.g. What breed of dog is this?",
                key="question_input"
            )

            if st.button("🔍 Get Answer", key="vqa_btn", use_container_width=True):
                if question.strip() == "":
                    st.warning("⚠️ Please enter a question first!")
                else:
                    with st.spinner("🤖 BLIP is thinking..."):
                        try:
                            model = load_model()
                            pil_image = preprocess_image(temp_path)
                            answer = model.answer_question(pil_image, question)

                            st.markdown(
                                f'<div class="answer-box">'
                                f'❓ <strong>Q:</strong> {question}<br><br>'
                                f'✅ <strong>A:</strong> {answer}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            # Save result
                            result = {
                                "task": "vqa",
                                "image": uploaded_file.name,
                                "question": question,
                                "answer": answer
                            }
                            os.makedirs("outputs", exist_ok=True)
                            with open("outputs/streamlit_results.json", "w") as f:
                                json.dump(result, f, indent=2)

                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

            with st.expander("🧠 How it works behind the scenes"):
                st.markdown("Both the image and your question are fed into BLIP together. The model uses cross-attention to find the answer by connecting visual regions to your question words.")

# ── Footer ──
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem; margin-top: 3rem;'>
    Built with PyTorch · Hugging Face Transformers · Streamlit<br>
    Model: Salesforce/blip-image-captioning-base & blip-vqa-base
</div>
""", unsafe_allow_html=True)