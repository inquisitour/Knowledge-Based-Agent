import streamlit as st
from agent import ResponseAgent
from data_processing import DBops
import pandas as pd
import os

def main():
    st.set_page_config(page_title="Document Genie", layout="wide")
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
        response = agent.answer_question(user_question)
        st.write("Response:", response)

if __name__ == "__main__":
    main()
