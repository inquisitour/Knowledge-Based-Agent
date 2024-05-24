import streamlit as st
import pandas as pd
import requests
from streamlit_lottie import st_lottie

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from agents.response_agent import ResponseAgent
from parent_agent import AgentState
from agents.utils_agent import UtilsAgent

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.set_page_config(page_title="Ophthal Agent", layout="wide")

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

    uploaded_csv = st.file_uploader("Upload your CSV data file", type=['csv'], key='csv')

    if uploaded_csv is not None:
        data_csv = pd.read_csv(uploaded_csv)
        # Pass the CSV data to the relevant agent for processing
        # For example: database_agent.process_local_file(data_csv)

    openai_api_key = UtilsAgent.get_env_variable("OPENAI_API_KEY")

    user_question = st.text_input("Enter a question:", key="user_question")
    if user_question:
        with st.spinner('Processing...'):
            response = AgentState(openai_api_key,"memory_test.db")
            st.write("Response:", response)

    lottie_url = 'https://assets1.lottiefiles.com/packages/lf20_vykpwt8b.json'
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        with st.container():
            st_lottie(lottie_json, height=200, width=300)

if __name__ == "__main__":
    main()