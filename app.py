import streamlit as st
import google.generativeai as genai 
from PIL import Image

# Configure Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load the Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Define function to generate response
def get_gemini_response(image): 
    # Define a fixed prompt template to generate captions for the uploaded image
    prompt = "Generate a single, detailed caption or description for the following image, in a natural and coherent paragraph format:"
    response = model.generate_content([prompt, image]) 
    return response.text

# Initialize the Streamlit app
st.set_page_config(page_title="Gemini Image Demo")
st.header("Caption Generator using Gemini API")

# File uploader for image input
uploaded_file = st.file_uploader("Choose any file :) ", type=["jpg", "jpeg", "png"])
image = ""

# Logic to display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image Uploaded.", use_column_width=True)

# Button to generate caption
submit = st.button("Generate Caption about image")

# If button clicked
if submit and image:
    response = get_gemini_response(image)
    # Display the generated caption
    st.subheader("Generated Caption is")
    st.write(response)
