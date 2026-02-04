import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("YOLO11 Object Detection")

# Load model
model = YOLO('yolo11n.pt')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['webp', 'jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert file to Image
    img = Image.open(uploaded_file)
    
    # Run inference
    results = model(img)
    
    # Show results
    # results[0].plot() returns a BGR numpy array
    res_plotted = results[0].plot()
    
    # Display the image
    st.image(res_plotted, caption='Detected Objects', use_container_width=True)
