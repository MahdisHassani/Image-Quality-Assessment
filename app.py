import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from models.model import get_model
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

st.title("📸 Image Quality Assessment")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", width=300)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor).item()

    st.write(f"### Predicted Quality Score: {pred:.3f}")

    if pred > 0.7:
        st.success("High Quality")
    elif pred > 0.4:
        st.warning("Medium Quality")
    else:
        st.error("Low Quality")