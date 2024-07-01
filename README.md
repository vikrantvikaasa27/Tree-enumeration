# Tree Enumeration and Green Cover Estimation

This project aims to provide an automated solution for tree enumeration and green cover estimation using satellite imagery and deep learning techniques. It leverages the YOLOv5 object detection model to detect trees in satellite images and estimates the green cover percentage in a given area.

## Features

- **Tree Counting**: Utilizes the YOLOv5 object detection model to accurately count the number of trees in a satellite image.
- **Green Cover Estimation**: Estimates the percentage of green cover in the specified area by analyzing the processed satellite image.
- **Interactive Interface**: Provides an interactive web application powered by Streamlit for users to upload satellite images and visualize the results.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Shobika-k2004/Tree-Enumeration.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run main.py
    ```

2. Upload a satellite image using the provided interface.
3. View the predicted tree count, processed image, and green cover percentage.

## Example
