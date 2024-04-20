import streamlit as st
from inference_engine import ResponseAgent
from data_preprocessing import DBops
import pandas as pd
import os

def main():
    st.set_page_config(page_title="Document Genie", layout="wide")
    st.title("Knowledge Based Agent")

    # Use an environment variable or a secure method to get the path
    file_path = os.getenv("DATA_FILE_PATH", "your_default_data_file.csv")
    
    if not DBops.check_data_hash(file_path):
        data = pd.read_csv(file_path)
        questions, answers = data['question'], data['answer']
        response_agent = ResponseAgent()
        embeddings = response_agent.embeddings.embed_documents(questions.tolist())  # Embeddings are generated here
        DBops.insert_data(questions, answers, embeddings)
        DBops.update_data_hash(file_path)

    user_question = st.text_input("Enter a question:", key="user_question")
    if user_question:
        response = ResponseAgent().answer_question(user_question)
        st.write("Response:", response)

if __name__ == "__main__":
    main()
