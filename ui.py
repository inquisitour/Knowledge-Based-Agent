import streamlit as st
from inference_engine import OpenAIops
from data_preprocessing import calculate_file_hash, DBops
import pandas as pd

def main():
    st.set_page_config(page_title="Document Genie", layout="wide")
    st.title("Gravitas Devel 1.0 - Advanced RAG System")
    file_path = "your_data_file.csv"  # Replace with your actual file path
    file_hash = calculate_file_hash(file_path)

    if not DBops.check_data_hash(file_hash):
        data = pd.read_csv(file_path)
        questions, answers = data['question'], data['answer']
        embeddings = OpenAIops.get_embeddings(questions.tolist())
        DBops.insert_data(questions, answers, embeddings)
        DBops.update_data_hash(file_hash)

    user_question = st.text_input("Enter a question?", key="user_question")
    if user_question:
        response = OpenAIops.answer_question(user_question)
        st.write("Response:", response)

if __name__ == "__main__":
    main()
