import time
import fitz
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import cv2
import pytesseract
import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import io
from sentence_transformers import SentenceTransformer, util

# PDF to Image conversion
def pdf_to_img(path):
    # Open the PDF file
    doc = fitz.open(path)
    
    # Set the desired resolution (e.g., 300 DPI)
    zoom_x = 300.0 / 72.0  # Horizontal zoom
    zoom_y = 300.0 / 72.0  # Vertical zoom
    
    for page in doc:
        try:
            # Create a pixmap from the page with the desired resolution
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
    
            # Create an image from the pixmap
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            image_path = "page.jpg"
            # Save the image to file
            img.save(image_path, "JPEG")
        except Exception as e:
            print(f"An error occurred while processing page {page.number}: {e}")

# Function to extract text from a given image after thresholding
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale conversion
    # Apply thresholding for better OCR
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresholded)

# Function to extract text from red, yellow, and blue highlighted regions in an image
def extract_text_from_highlighted_regions(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image from {image_path}.")
        return None

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, and blue
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])

    # Create separate masks for each color
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Isolate highlighted regions for each color
    red_highlighted = cv2.bitwise_and(image, image, mask=red_mask)
    yellow_highlighted = cv2.bitwise_and(image, image, mask=yellow_mask)
    blue_highlighted = cv2.bitwise_and(image, image, mask=blue_mask)

    # Extract text from each color-highlighted region
    red_text = extract_text_from_image(red_highlighted)
    yellow_text = extract_text_from_image(yellow_highlighted)
    blue_text = extract_text_from_image(blue_highlighted)

    # Return dictionary with the extracted text for each color
    return {
        "red": red_text,
        "yellow": yellow_text,
        "blue": blue_text,
    }
# Function to calculate similarity between highlighted text and a reference text

def compare_dict_with_file(dict1, file_path):

    # Load pre-trained DistilBERT model
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    # Create a dictionary from the file
    dict2 = {}
    with open(file_path, "r") as file:
        # Loop through each line and extract key-value pairs
        for line in file:
            if ":" in line:
                key, value = line.split(":", 1)
                dict2[key.strip()] = value.strip()

    # Compute the similarity for each key
    similarities = {}
    for key in dict1:
        if key in dict2:
            # Generate embeddings for the text
            embedding1 = model.encode(dict1[key])
            embedding2 = model.encode(dict2[key])

            # Compute cosine similarity
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

            # Store the similarity score
            similarities[key] = similarity
        else:
            similarities[key] = None  # If the key does not exist in dict2

    return similarities

# Extract image from PDF
def extract_images(pdf_path):
    pdf_document = fitz.open(pdf_path)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)

        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption=f"Page {page_number+1}, Image {image_index}")

    pdf_document.close()


if __name__ == "__main__":
    #pytesseract.pytesseract.tesseract_cmd = r"C:/Users/vinay/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0/LocalCache/local-packages/Python310/site-packages/Tesseract-OCR/tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR/tesseract.exe"
    start = time.perf_counter()
    print("Program start: ", start)

    
    st.set_page_config(page_title="Text Extraction Task",
                    page_icon='°□°',
                    layout='centered',
                    initial_sidebar_state='collapsed')
    st.header("Text Extraction ")
    col1,col2 = st.columns([10,10])

    #For upload option from Streamlit UI

    ##with col1:
    ##    pdf_f"ile = st.file_uploader("Upload PDF file", type="pdf", key=1)

    # Read PDF file
    pdf_file = "The Dynamics of Power Grids - chatGPT.pdf"
    

    #with col2: 
    #    text_file = st.file_uploader("Upload text file", type="txt", key=2)
    
    #col3,col4 = st.columns([10,10])

    #with col3:
    if pdf_file is not None:
        print(pdf_file)
        page = pdf_to_img(pdf_file)
        image_path = "page.jpg"
        reference_text_path = "notes.txt"
        highlighted_texts = extract_text_from_highlighted_regions(image_path)
        print(highlighted_texts)
        extract_images(pdf_file)
        st.write(highlighted_texts)
        similarity_results = compare_dict_with_file(highlighted_texts, reference_text_path )
        # Display the results
        for color_name, similarity_percentage in similarity_results.items():
            print(f"Similarity for {color_name} highlights: {similarity_percentage*100:.2f}%")
            st.write(f"Similarity for {color_name} highlights: {similarity_percentage*100:.2f}%")
        # st.image(page, caption="This is my image")
    


    end = time.perf_counter()
    print("Program end: ", end)