import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Crack Detection App", layout="centered")

st.title("AI-Based Pipeline Structural Health Monitoring System")

st.write("Upload an image to detect cracks in structures or pipelines.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Error reading image. Please upload a valid image.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blur, 10, 60)

        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Overlay cracks (red color)
        result = image_rgb.copy()
        result[edges > 0] = [255, 0, 0]

        # Display images
        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image_rgb, caption="Original Image")

        with col2:
            st.image(result, caption="Detected Cracks (Highlighted)")

        # Detection message
        if np.sum(edges) > 0:
            st.success("Crack Detected")
        else:
            st.info("No Crack Detected")

        # Download button
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        st.download_button(
            label="Download Result Image",
            data=buffer.tobytes(),
            file_name="crack_result.jpg",
            mime="image/jpeg"
        )