import streamlit as st
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# Initialize translation model
model_name = 'Helsinki-NLP/opus-mt-en-de'  # Example model for English to German translation
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

st.title('Language Detection and Translation System')

# Language Detection
text = st.text_input('Enter text for language detection:')
detected_lang = detect(text)
st.write(f'Detected Language: {detected_lang}')

# Translation
if st.button('Translate'):
    translated_text = model.generate(**tokenizer(text, return_tensors="pt")[0], max_length=100)[0]
    translated_text = tokenizer.decode(translated_text, skip_special_tokens=True)
    st.write(f'Translated Text: {translated_text}')

if __name__ == '__main__':
    st.write('Language Detection and Translation System is running...')
