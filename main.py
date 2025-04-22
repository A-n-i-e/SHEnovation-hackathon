import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
# import sys

def configure_gemini():
    #this is to load the api key and configute the gemini client
    load_dotenv()
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        print("GEMINI_KEY not found in .env file")
    
    genai.configure(api_key=api_key)

def perform_ocr(image, prompt):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    print("Sending request to Gemini API...")

    try:
        response = model.generate_content([prompt, image])

        extracted_text = response.text
        print("Response received from Gemini API.")
        return extracted_text
    
    except Exception as e:
        return f"An error has occurred during Gemini API call: {e}"
    

image = "poem 2.jpg"
image = Image.open(image)
print
prompt = """Perform Optical Character Recognition (OCR) on the provided image.
        Extract all visible text accurately.
        Preserve line breaks and general layout if possible.
        Output *only* the recognized text, without any additional comments, descriptions, or summaries.
        Only read the text if its part of a full page in the book.
    """

configure_gemini()

result = perform_ocr(image, prompt)
print("Extracted Text:")
print(result)
print("-----------------")