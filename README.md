# Text-Image-Extraction-from-PDF

The script extracts highlighted text from a PDF document and compares it with text from a `notes.txt` file for each color of highlighting. The program utilizes an open-source Large Language Model (LLM) - Distil Roberta for text comparison and returns the similarity percentage (ranging from 0 to 100%) for the text corresponding to each color.

The image from the PDF is extracted is displayed in the streamlit UI along with the text that is highlighted in the PDF and the similarity scores.

## Tech
Script uses the following tech stack:

- [Python3.x] - An interpreted high-level general-purpose programming language.
- [Streamlit](https://streamlit.io/) - A framework for building interactive web apps with Python.
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) - A powerful open-source Optical Character Recognition (OCR) engine.
- [Transformers](https://huggingface.co/transformers/) - A library for working with large language models like Distil Roberta.


## Installation

- Scripts used: **streamer.py**
- Place the downloaded repository into a directory of choice.
- Download the "**Tesseract** exe file from the link : https://github.com/UB-Mannheim/tesseract/wiki. Place the Tesseract-OCR folder in the working directory
- Navigate to the working directory and run the following:

```
pip install -r requirements.txt

```
or 

```
pip install the modules mentioned in the requirements.txt file

```
## Usage

- Open a terminal and navigate to the project directory.
- Start the Streamlit app with the following command:
-    **streamlit run .\streamer.py**
- Access the Streamlit web app through the provided URL, typically http://localhost:8501/.
- The web app will display the extracted text and similarity scores for each highlighted section along with the extracted image.

## Troubleshooting

If you encounter any issues while using the application, check the following common solutions:

1. **Tesseract OCR Issues**:
   - Ensure that Tesseract OCR is installed correctly. You can download it from [here](https://github.com/UB-Mannheim/tesseract/wiki).
   - Make sure the `Tesseract-OCR` folder is in your system's PATH. If not, add it manually or place the folder in the project directory.
   - If the text extraction from the PDF seems incorrect, double-check that the PDF file is valid and contains the highlighted text.

2. **Streamlit Issues**:
   - Check the terminal where you started the Streamlit app for error messages.
   - Ensure all required Python packages are installed. You can reinstall them with:
     ```shell
     pip install -r requirements.txt
     ```
   - If the Streamlit app doesn't start or behaves unexpectedly, try restarting it or checking for updates to Streamlit and related packages.

3. **Similarity Score Issues**:
   - If the similarity scores seem off or incorrect, ensure that the `notes.txt` file contains the expected text for comparison.
   - Verify that the highlighted text in the PDF aligns with the content in the `notes.txt` file.
   - If you're seeing low similarity scores, check if the text has formatting issues or is too brief for a meaningful comparison.


