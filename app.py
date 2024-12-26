import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set Streamlit page configuration
def set_page_config():
    st.set_page_config(
        page_title='Caption an Image', 
        page_icon=':camera:', 
        layout='wide'
    )

# Initialize the BLIP model and processor
@st.cache_resource
def initialize_model():
    hf_model = "Salesforce/blip-image-captioning-large"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = BlipProcessor.from_pretrained(hf_model)
    model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)
    return processor, model, device

# Upload an image from the sidebar
def upload_image():
    return st.sidebar.file_uploader("Upload an image (we aren't storing anything)", type=["jpg", "jpeg", "png"])

# Resize image for display (optional max width)
def resize_image(image, max_width=700):
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        height = int(height * ratio)
        image = image.resize((max_width, height))
    return image

# Generate a caption for the uploaded image
def generate_caption(processor, model, device, image):
    inputs = processor(image, return_tensors='pt').to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Main function to run the Streamlit app
def main():
    set_page_config()
    st.header("Caption an Image :camera:")

    # Upload image
    uploaded_image = upload_image()

    # If an image is uploaded, process and display it
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = resize_image(image)  # Resize image for better display

        # Display the uploaded image
        st.image(image, caption='Your image')

        # Sidebar section for generating caption
        with st.sidebar:
            st.divider()  # Divider for aesthetic spacing
            if st.button('Generate Caption'):
                # Display loading spinner while generating the caption
                with st.spinner('Generating caption...'):
                    processor, model, device = initialize_model()
                    caption = generate_caption(processor, model, device, image)
                    st.header("Generated Caption:")
                    st.markdown(f'**{caption}**')

if __name__ == '__main__':
    main()
