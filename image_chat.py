import streamlit as st
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import gc

st.set_page_config(page_title="LLaVA Vision Demo", layout="centered")
st.title("🦙 LLaVA 1.5 - Vision-Language Model Demo")
st.write("Upload an image and enter a prompt. The model will generate a description or answer.")

@st.cache_resource()
def load_model():
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

processor, model = load_model()

# --- Session state setup ---
if "result" not in st.session_state:
    st.session_state.result = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

uploaded_image = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    st.session_state.uploaded_image = uploaded_image

prompt_text = st.text_area("💡 Enter your prompt (e.g., *Describe this image in detail.*)", height=100)

if st.button("Generate Response"):
    if st.session_state.uploaded_image and prompt_text:
        with st.spinner("Generating response, please wait..."):
            image = Image.open(st.session_state.uploaded_image).convert("RGB")
            prompt = "<image>\n" + prompt_text

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )

            result = processor.decode(output[0], skip_special_tokens=True)
            st.session_state.result = (image, result)

            # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()
    else:
        st.warning("⚠ Please upload an image and enter a prompt before generating.")

# --- Display result if present in session state ---
if st.session_state.result:
    img, output_text = st.session_state.result
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("### 🤖 Model Response:\n")
    st.markdown(f"```\n{output_text}\n```")
