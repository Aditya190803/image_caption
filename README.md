# Image Caption Generator
This Streamlit app allows users to upload an image, and it automatically generates a caption or description for the image using the Gemini API. The app uses Google's **Generative AI (Gemini)** model to process the image and generate a detailed, coherent caption in a natural paragraph format.

## Features
- Upload an image in formats such as JPG, JPEG, or PNG.
- Generate a detailed caption describing the uploaded image.
- The caption is generated using the **Gemini API** by Google.

## Prerequisites
Before running the app, ensure you have the following:
- Python 3.7 or later.
- Streamlit installed.
- Google Generative AI library (gemini) installed.
- An API key from Google Gemini API.

## Installation

### Step 1: Install Dependencies
You need to install Streamlit and the necessary libraries. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### Step 2: API Key Configuration
To access the Gemini API, you will need a valid API key from Google. Store this API key securely.

Create a `.streamlit/secrets.toml` file in the project directory and add your Gemini API key like this:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

### Step 3: Run the App
After installing the dependencies and configuring the API key, run the following command to start the Streamlit app:

```bash
streamlit run app.py
```

### Step 4: Interact with the App
1. Once the app is running, you will be prompted to upload an image file (JPG, JPEG, or PNG).
2. After uploading the image, click on the "Generate Caption" button to receive a caption describing the image.
3. The app will display the generated caption below the uploaded image.

## How it Works

The app uses the following steps:
1. **Image Upload**: The user uploads an image file through the Streamlit UI.
2. **Caption Generation**: Upon pressing the "Generate Caption" button, the app sends the image and a prompt to the Gemini API, requesting it to generate a detailed caption or description.
3. **Display**: The generated caption is displayed on the page below the image.

### Key Components
- **`st.file_uploader`**: Streamlit component that allows users to upload files.
- **`genai.GenerativeModel`**: A class used to interact with Google's Gemini API, which processes the image and generates content.
- **`Image.open`**: A method from the PIL (Pillow) library to open and display the uploaded image.
- **API Integration**: The `get_gemini_response` function sends the image along with a prompt to the Gemini model to generate a caption.

## Troubleshooting

- **Gemini API Key Errors**: Make sure your API key is correctly configured in the `secrets.toml` file.
- **Image Upload Issues**: If the uploaded image doesn't show, check if the image file format is supported (JPG, JPEG, PNG).
- **Generation Issues**: If the caption generation fails, verify your internet connection and ensure the Gemini API key is valid.