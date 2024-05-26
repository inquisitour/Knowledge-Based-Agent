import os
import requests
import pandas as pd
from typing import Dict, Any
from parent_agent import ParentAgent

import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

import streamlit as st
from streamlit_lottie import st_lottie

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_state(state: Dict[str, Any], uploaded_csv=None, user_query=None) -> Dict[str, Any]:
    if uploaded_csv is not None:
        data_csv = pd.read_csv(uploaded_csv)
        state["data_csv"] = data_csv.to_dict(orient="list")
    if user_query is not None:
        state["user_query"] = user_query
    return state

def main():
    st.set_page_config(page_title="Knowledge Based Agent", layout="wide")

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
    user_query = st.text_input("Enter a question:", key="user_question")

    # Initialize the state with default values
    initial_state: Dict[str, Any] = {
        "user_query": "",
        "database_retrieval": [],
        "graph_retrieval": {},
        "context_combination": {},
        "data_csv": {}
    }
    
    # Setup agent memory and instantiate
    db_path = os.path.join(ROOT_DIR, 'agents', 'memory', 'agent_session.db')
    init_state = load_state(state=initial_state, uploaded_csv=uploaded_csv, user_query=user_query)

    if init_state["data_csv"] and init_state["user_query"]:
        with st.spinner('Processing...'):
            response = ParentAgent(db_path, init_state)
            st.write("Response:", response)

    lottie_url = 'https://assets1.lottiefiles.com/packages/lf20_vykpwt8b.json'
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        with st.container():
            st_lottie(lottie_json, height=200, width=300)

if __name__ == "__main__":
    main()