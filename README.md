# 🦙 LLaVA 1.5 Vision-Language Demo Repository

This repository contains an end-to-end implementation of a vision-language inference demo using the open-source **LLaVA 1.5 (7B)** model, deployed through a **Streamlit** interface. The project was used for a live demo showcasing how to process an image + prompt input and generate descriptive or conversational responses.

---

## 📂 Repository Structure

```
llava-vision-demo/
│
├── image_chat.py                # Python script containing inference logic wrapped with Streamlit 
├── Demo.ipynb                   # Jupyter notebook explaining the full pipeline
├── requirements.txt             # Required dependencies
├── README.md                    # Detailed explanation of the repository
```

---

## ✅ About the Code

### 1. image\_chat.py

This is the final Python script that wraps the entire content and logic from **Demo.ipynb** using the Streamlit library and exposes it as an interactive API for end users. The following are the key components:

- Model loading from Hugging Face
- Image upload and prompt input handling
- Input processing through the processor
- Inference execution on GPU with float16 precision
- Response decoding and display in Streamlit
- Includes memory cleanup and session state management for smooth operation

This script serves as the production-ready interface for demonstrating vision-language inference and allows for seamless user interaction during the demo.

### 2. Demo.ipynb

- A detailed Jupyter notebook that walks through:
  - Downloading model weights from Hugging Face
  - Preparing input images and prompts for inference
  - Step-by-step explanation of how the processor and model work together
  - Example inference runs with visual and text outputs for clarity

### 3. requirements.txt 

- Contains all necessary dependencies including:
  - `transformers` (latest from Hugging Face GitHub)
  - `torch`, `accelerate`, `bitsandbytes`, `pillow`, `streamlit`, `sentencepiece`, and `einops`

### 4. **Environment & Deployment Setup**

For this demo, I chose to use a **Jarvislabs.ai RTX5000 instance**, which comes with:

- 16GB GPU memory
- 32GB RAM
- 20GB storage

This platform was selected for its quick setup and easy instance creation, making deployment smooth and hassle-free. If you’d like to explore JarvisLabs, you can visit their website at [https://jarvislabs.ai](https://jarvislabs.ai).

---

## 🎯 About the Demo

- The demo takes **both an image and a text prompt** as input.
- Uses **LLaVA 1.5 (7B)** from Hugging Face.
- The model generates descriptive or conversational responses.
- Tested on educational images, posters, and real-life photographs.

### ✅ Sample Prompts for Demo:

- "Describe this image in detail."
- "List the animals shown in this image."
- "What category of animals are depicted?"
- "Write a creative caption for this image."

### ✅ Key Features Highlighted During the Demo:

- Efficient GPU inference
- Smooth memory management
- Practical fallback for long-running sessions
- Easy user interface built with Streamlit

---

## ✅ Model Used

- **LLaVA 1.5 (7B)** — Vision-language model available on Hugging Face:\
  [https://huggingface.co/llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)

---

## 📬 Questions or collaboration?

Feel free to raise issues or reach out if you’d like to replicate this setup or integrate it into your product!

> *Built and demoed with the goal of making vision-language inference accessible and practical.*

