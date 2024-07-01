import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

# Load the trained YOLOv5 model
model = YOLO(r"C:\Projects\tree enumeration\Models\best.pt")

# Function to process the uploaded image and estimate green cover and tree count
def process_image(image_bytes):
    # Convert BytesIO object to OpenCV format
    image_cv2 = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

    # Make predictions on the input image
    predictions = model(image_cv2)

    # Count the number of trees detected
    tree_count = sum(len(pred) for pred in predictions)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the percentage of green cover and idle land
    total_pixels = np.prod(binary.shape[:2])
    green_pixels = total_pixels - cv2.countNonZero(binary)
    green_cover_percentage = (green_pixels / total_pixels) * 100
    idle_land_percentage = 100 - green_cover_percentage

    return tree_count, binary, green_cover_percentage, idle_land_percentage

# Streamlit app
def main():
    st.title("Tree Enumeration and Green Cover Estimator")

    # File uploader for image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Process the image to estimate tree count, green cover, and idle land
        tree_count, binary, green_cover_percentage, idle_land_percentage = process_image(uploaded_image)

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Display the tree count
        st.subheader("Tree Count:")
        st.write('Predicted Tree Count:  ',str(tree_count))

        # Display the processed binary mask of the program of the pe
        st.image(binary, caption="Processed Image", use_column_width=True)

        # Display the green cover percentage
        st.subheader("Green Cover Percentage:")
        st.write(f"{green_cover_percentage:.2f}%")

        # Display the idle land percentage
        st.subheader("Idle Land Percentage:")
        st.write(f"{idle_land_percentage:.2f}%")

if __name__ == "__main__":
    main()
