import os
import psycopg2
from psycopg2 import pool, extras
from hashlib import sha256
import boto3
import numpy as np
import pandas as pd
from io import BytesIO
from contextlib import contextmanager
from langchain.embeddings.openai import OpenAIEmbeddings

# Function to fetch database configuration from environment variables
def get_db_config():
    return {
        "user": os.getenv("DB_USER", "default_user"),
        "password": os.getenv("DB_PASSWORD", "default_pass"),
        "host": os.getenv("DB_HOST", "default_host"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "default_db")
    }

# Fetch database configuration
db_config = get_db_config()

# Initialize the connection pool with the fetched configuration
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=10,  # Adjust maxconn based on your expected workload
    **db_config
)

@contextmanager
def get_database_connection():
    conn = connection_pool.getconn()
    try:
        yield conn
    finally:
        connection_pool.putconn(conn)

def with_connection(func):
    def wrapper(*args, **kwargs):
        with get_database_connection() as conn:
            return func(*args, **kwargs, conn=conn)
    return wrapper

class DBops:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002")

    def calculate_file_hash(self, file_content):
        return sha256(file_content).hexdigest()

    def process_file_from_s3(self, bucket_name, file_key):
        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = obj['Body'].read()
        if not self.check_data_hash(file_hash):
            print("Data hash mismatch found. Updating database...")
            
            csv_data = pd.read_csv(BytesIO(file_content))
            if 'questions' in csv_data.columns and 'answers' in csv_data.columns:
                questions = csv_data['questions'].tolist()
                answers = csv_data['answers'].tolist()
                embeddings = self.embeddings.embed_documents(questions)  # Batch processing
                self.insert_data(questions, answers, embeddings)
                self.delete_all_data_hashes()
                self.update_data_hash(file_hash)
                print("Database updated with new data and data hash")
            else:
                raise ValueError("CSV does not contain the required 'questions' and 'answers' columns")
        else:
            print("Data is up to date")
    
    @with_connection
    def insert_data(self, questions, answers, embeddings, conn):
        print("Inserting data into database")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM faq_embeddings")
            args = [(q, a, psycopg2.Binary(np.array(emb).astype(np.float32).tobytes())) for q, a, emb in zip(questions, answers, embeddings)]
            extras.execute_batch(cur, "INSERT INTO faq_embeddings (question, answer, embedding) VALUES (%s, %s, %s)", args)
            conn.commit()

    @with_connection
    def check_data_hash(self, file_hash, conn):
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM data_hash WHERE file_hash = %s)", (file_hash,))
            return cur.fetchone()[0]

    @with_connection
    def update_data_hash(self, file_hash, conn):
        with conn.cursor() as cur:
            cur.execute("INSERT INTO data_hash (file_hash) VALUES (%s) ON CONFLICT (file_hash) DO NOTHING", (file_hash,))
            conn.commit()

    @with_connection
    def delete_all_data_hashes(self, conn):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM data_hash")  
            conn.commit()
            print("Deleted all data hashes from the database.")

    @with_connection
    def setup_database(self, conn):
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
                    file_hash TEXT UNIQUE
                );
            """)
            conn.commit()
            print("Database tables created or verified.")
