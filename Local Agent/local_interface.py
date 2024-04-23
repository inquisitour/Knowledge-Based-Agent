import streamlit as st
from agent import ResponseAgent
from data_processing import DBops
import pandas as pd
import os

# Import for Lottie animations
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.set_page_config(page_title="Document Genie", layout="wide")

    # Adding custom CSS for blurring input fields
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            filter: blur(3px);
        }
        .stTextInput > div > div > input:focus {
            filter: blur(0px);
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Knowledge Based Agent")

    # Ensure the database is setup before handling any events
    db_ops = DBops()
    db_ops.setup_database()

    uploaded_file = st.file_uploader("Upload your data file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        db_ops.process_local_file(data)
    
    agent = ResponseAgent()
    print("Agent initialised!")

    # User query section
    user_question = st.text_input("Enter a question:", key="user_question")
    if user_question:
        with st.spinner('Processing...'):
            response = agent.answer_question(user_question)
            st.write("Response:", response)

    # Adding a Lottie animation for loading
    lottie_url = 'https://assets1.lottiefiles.com/packages/lf20_vykpwt8b.json'
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        with st.container():
            st_lottie(lottie_json, height=200, width=300)

if __name__ == "__main__":
    main()
