import os
import pandas as pd
import psycopg2
from hashlib import sha256
from psycopg2 import OperationalError, Error

def get_database_url():
    """Retrieve the database URL from environment variables."""
    db_host = os.getenv("DB_HOST", "default_host")
    db_name = os.getenv("DB_NAME", "default_db")
    db_password = os.getenv("DB_PASSWORD", "default_pass")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "default_user")
    return f"postgres://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

DATABASE_URL = get_database_url()

class DBops:
    @staticmethod
    def setup_database():
        try:
            with psycopg2.connect(DATABASE_URL) as conn:
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
        except (OperationalError, Error) as e:
            print("Database operation failed:", e)

    @staticmethod
    def insert_data(questions, answers, embeddings):
        try:
            with psycopg2.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    for question, answer, embedding in zip(questions, answers, embeddings):
                        cur.execute(
                            "INSERT INTO faq_embeddings (question, answer, embedding) VALUES (%s, %s, %s)",
                            (question, answer, psycopg2.Binary(embedding))
                        )
                conn.commit()
        except (OperationalError, Error) as e:
            print("Database operation failed:", e)

    @staticmethod
    def check_data_hash(file_path):
        file_hash = DBops.calculate_file_hash(file_path)
        try:
            with psycopg2.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT file_hash FROM data_hash WHERE file_hash = %s;", (file_hash,))
                    return cur.fetchone() is not None
        except (OperationalError, Error) as e:
            print("Database operation failed:", e)
            return False

    @staticmethod
    def update_data_hash(file_path):
        file_hash = DBops.calculate_file_hash(file_path)
        try:
            with psycopg2.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO data_hash (file_hash) VALUES (%s);", (file_hash,))
                    conn.commit()
        except (OperationalError, Error) as e:
            print("Database operation failed:", e)

    @staticmethod
    def calculate_file_hash(file_path):
        try:
            with open(file_path, "rb") as f:
                file_hash = sha256()
                while chunk := f.read(4096):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except IOError as e:
            print(f"File operation failed: {e}")
            raise

    @staticmethod
    def get_similar_questions(embedding):
        try:
            with get_database_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT question, answer FROM faq_embeddings ORDER BY (embedding <-> %s) LIMIT 10;",
                        (psycopg2.Binary(embedding),)
                    )
                    return cur.fetchall()
        except (OperationalError, Error) as e:
            print("Database operation failed:", e)
            raise
    
    @staticmethod
    def get_database_connection():
    database_url = get_database_url()
    try:
        return psycopg2.connect(database_url)
    except (OperationalError, Error) as e:
        print("Error connecting to the database:", e)
        raise
