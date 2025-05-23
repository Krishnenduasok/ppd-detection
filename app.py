import streamlit as st
import torch
import asyncio
from torchvision import transforms
from PIL import Image
import os
from model import ResNet50WithDropout
import requests
from io import BytesIO

# --- Fix for "RuntimeError: no running event loop" (Python 3.12 compatibility) ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Class labels ---
class_names = ['high', 'low', 'md', 'medium', 'zero']

# --- Load model from Google Drive ---
@st.cache_resource
def load_model():
    try:
        file_id = "11kWodly2XNUcOt7HdlBbD77sTsC4_I9o"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"

        st.info("🔄 Downloading model weights from Google Drive...")
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("⚠️ Failed to download model from Google Drive. Check file permissions or URL.")

        buffer = BytesIO(response.content)

        model = ResNet50WithDropout(num_classes=len(class_names))
        model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
        model.eval()
        st.success("✅ Model loaded successfully.")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# --- Image transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- App Title ---
st.markdown("<h1 style='color:#198754;'>🧪 PPD Score Prediction from Tuber Images of Cassava</h1>", unsafe_allow_html=True)
st.subheader("Upload a cassava tuber image to predict the Postharvest Physiological Deterioration (PPD) score.")

# --- File uploader ---
uploaded_file = st.file_uploader("📤 Upload a cassava tuber image...", type=["jpg", "jpeg", "png"])

# --- Sidebar Examples ---
st.sidebar.markdown("### 📊 Example Class Images")
example_folder = "examples"
for class_name in class_names:
    class_path = os.path.join(example_folder, class_name)
    if os.path.isdir(class_path):
        st.sidebar.markdown(f"**{class_name.upper()}**")
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        for img_file in image_files[:2]:  # show up to 2 per class
            img_path = os.path.join(class_path, img_file)
            st.sidebar.image(img_path, use_container_width=False, width=160)
    else:
        st.sidebar.warning(f"⚠️ No folder found for '{class_name}' in examples/")

# --- Prediction Section ---
if uploaded_file is not None:
    if model is None:
        st.error("❌ Model failed to load. Please check model definition or link.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]

        st.success(f"✅ Predicted Class: **{predicted_class.upper()}**")
