import Streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Supported languages and their MarianMT model names
languages = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Hindi': 'hi'
}

# Initialize language models
def get_model(language_code):
    model_name = f'Helsinki-NLP/opus-mt-en-{language_code}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Function to translate text
def translate_text(text, dest_language_code):
    tokenizer, model = get_model(dest_language_code)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

# App title
st.title("Patient Registration Form")

# Select language
selected_language = st.selectbox("Choose Language", list(languages.keys()))
dest_language_code = languages[selected_language]

# Patient Registration Form
st.header(translate_text("Personal Information", dest_language_code))
name = st.text_input(translate_text("Name", dest_language_code))
age = st.number_input(translate_text("Age", dest_language_code), min_value=0)
gender = st.selectbox(translate_text("Gender", dest_language_code), [translate_text("Male", dest_language_code), translate_text("Female", dest_language_code)])
contact = st.text_input(translate_text("Contact Number", dest_language_code))

st.header(translate_text("Medical Information", dest_language_code))
symptoms = st.text_area(translate_text("Symptoms", dest_language_code))
previous_conditions = st.text_area(translate_text("Previous Medical Conditions", dest_language_code))

# Submit button
if st.button(translate_text("Submit", dest_language_code)):
    st.success(translate_text("Form submitted successfully!", dest_language_code))
