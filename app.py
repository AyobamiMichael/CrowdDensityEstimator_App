import streamlit as st
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torch
import numpy as np

# Set page config
st.set_page_config(page_title="Face Counter & Density Estimator", layout="wide")

# Load MTCNN model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# App title and description
st.markdown("""
    <h1 style='text-align: center;'>🎯 Face Counter & Density Estimator</h1>
    <p style='text-align: center; font-size: 18px;'>
        Upload an image or use your webcam to detect faces and estimate crowd density using MTCNN.
    </p>
""", unsafe_allow_html=True)

# Sidebar - Upload or Webcam
st.sidebar.title("Input Options")
input_type = st.sidebar.radio("Choose input type:", ("Upload Image", "Use Webcam"))

image = None
if input_type == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif input_type == "Use Webcam":
    cam_image = st.camera_input("Take a picture")
    if cam_image:
        image = Image.open(cam_image).convert("RGB")

# If image is available
if image:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    # Detect faces
    boxes, _ = mtcnn.detect(image)
    face_count = 0 if boxes is None else len(boxes)

    # Classify density
    if face_count <= 10:
        density = "🟢 Sparse"
    elif face_count <= 50:
        density = "🟡 Medium"
    else:
        density = "🔴 Dense"

    # Annotate image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline="red", width=3)

    with col2:
        st.image(annotated, caption="Detected Faces", use_column_width=True)

    st.markdown(f"""
    <hr>
    <div style='text-align: center;'>
        <h2>🧮 Face Count: <span style='color: blue;'>{face_count}</span></h2>
        <h2>📊 Crowd Density: <span style='color: green;'>{density}</span></h2>
    </div>
    <hr>
    """, unsafe_allow_html=True)

else:
    st.info("Please upload an image or take a photo using your webcam.")
