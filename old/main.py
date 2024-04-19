import os
import pandas as pd
import psycopg2
import hashlib
from psycopg2 import OperationalError, Error
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SimpleRetriever  #can use custom retrievers here

# Set environment variables (ensure to replace with your actual key and database credentials)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
DATABASE_URL = "postgres://username:password@hostname:port/dbname"

# Establish database connection
try:
    conn = psycopg2.connect(DATABASE_URL)
except OperationalError as e:
    st.error(f"Failed to connect to the database: {e}")
    raise e

class OpenAIops:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    chat_model = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-3.5-turbo'
    )
    retriever = SimpleRetriever(embeddings)

    @staticmethod
    def get_embeddings(questions):
        return OpenAIops.embeddings.embed_documents(questions)

    @staticmethod
    def get_context(rows):
        documents = OpenAIops.retriever.retrieve(rows)
        context = "\n".join([f"Question: {doc['content'][0]}\nAnswer: {doc['content'][1]}" for doc in documents])
        return context
        #return "\n".join([f"Question: {row[0]}\nAnswer: {row[1]}" for row in rows])

    @staticmethod
    def answer_question(user_question):
        try:
            user_embedding = OpenAIops.get_embeddings([user_question])[0]
            similar_questions = DBops.get_similar_questions(user_embedding)
            context = OpenAIops.get_context(similar_questions)

            prompt_template = (
                "Answer the question as detailed as possible from the provided context which can be info from "
                "the questions or the answers, make sure to provide all the details, account for spelling errors assume "
                "the closest meaning of the question, if information about the question or any similar question is not in provided context "
                "just say, 'answer is not available in the context', don't provide the wrong answer\n\n"
                f"Context:\n{context}\n\nQuestion: \n{user_question}\nAnswer:"
            )

            sysmsg1 = (
                "You are a helpful assistant. You will be given context and based on that you have to answer questions in "
                "every prompt. Don't provide an answer if the information is not in the context."
            )

            sysmsg2 = (
                "Develop a Retrieval-Augmented Generation (RAG) system that uses a structured question-answer database as its context. "
                "The system should:\n\n"
                "Input Processing: Accept a user question and preprocess it to correct any spelling errors and clarify ambiguous terms.\n"
                "Contextual Retrieval: Search the question-answer database to find question-answer pairs that are most relevant to the "
                "processed user question. Utilize natural language processing techniques to match the semantics of the question rather than "
                "relying solely on keyword matching.\n"
                "Answer Generation:\n"
                "    If relevant information is available: Use the retrieved question-answer pairs to generate a comprehensive and detailed response. "
                "The answer should integrate all relevant information from the context, ensuring that it addresses all aspects of the user's question. "
                "The system should synthesize the information in a coherent and informative manner.\n"
                "    If no relevant information is available: The system should return 'Answer not available in the context' to indicate that it "
                "cannot provide an accurate answer based on the existing database.\n"
                "Output: Present the answer to the user in a clear and concise format. If multiple question-answer pairs are relevant, synthesize the "
                "information into a single unified response to avoid redundancy and ensure clarity."
            )

            response = OpenAIops.chat_model.generate_response([sysmsg1, sysmsg2, prompt_template])
            return response
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")
            return "An error occurred while generating the answer."

class DBops:
    @staticmethod
    def setup_database():
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS faq_embeddings (
                    id SERIAL PRIMARY KEY,
                    question TEXT,
                    answer TEXT,
                    embedding BYTEA
                );
                CREATE TABLE IF NOT EXISTS data_hash (
                    id SERIAL PRIMARY KEY,
                    file_hash TEXT
                );
            """)
            conn.commit()

    @staticmethod
    def insert_data(questions, answers, embeddings):
        with conn.cursor() as cur:
            for question, answer, embedding in zip(questions, answers, embeddings):
                cur.execute("INSERT INTO faq_embeddings (question, answer, embedding) VALUES (%s, %s, %s)",
                            (question, answer, psycopg2.Binary(embedding)))
            conn.commit()

    @staticmethod
    def get_similar_questions(embedding):
        with conn.cursor() as cur:
            cur.execute("""
                SELECT question, answer FROM faq_embeddings
                ORDER BY (embedding <-> %s) LIMIT 10;
            """, (psycopg2.Binary(embedding),))
            return cur.fetchall()

    @staticmethod
    def check_data_hash(file_hash):
        with conn.cursor() as cur:
            cur.execute("SELECT file_hash FROM data_hash ORDER BY id DESC LIMIT 1;")
            last_hash = cur.fetchone()
            return last_hash == (file_hash,)

    @staticmethod
    def update_data_hash(file_hash):
        with conn.cursor() as cur:
            cur.execute("INSERT INTO data_hash (file_hash) VALUES (%s);", (file_hash,))
            conn.commit()

def calculate_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(4096):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def main():
    st.set_page_config(page_title="Document Genie", layout="wide")
    st.title("Gravitas Devel 1.0 - Advanced RAG System")
    DBops.setup_database()

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
