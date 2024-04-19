import pandas as pd
import psycopg2
import hashlib
from psycopg2 import OperationalError, Error

DATABASE_URL = "postgres://username:password@hostname:port/dbname"

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
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                for question, answer, embedding in zip(questions, answers, embeddings):
                    cur.execute(
                        "INSERT INTO faq_embeddings (question, answer, embedding) VALUES (%s, %s, %s)",
                        (question, answer, psycopg2.Binary(embedding))
                    )
            conn.commit()

    @staticmethod
    def get_similar_questions(embedding):
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT question, answer FROM faq_embeddings ORDER BY (embedding <-> %s) LIMIT 10;",
                    (psycopg2.Binary(embedding),)
                )
                return cur.fetchall()

    @staticmethod
    def check_data_hash(file_hash):
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT file_hash FROM data_hash ORDER BY id DESC LIMIT 1;")
                last_hash = cur.fetchone()
                return last_hash == (file_hash,)

    @staticmethod
    def update_data_hash(file_hash):
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO data_hash (file_hash) VALUES (%s);", (file_hash,))
                conn.commit()

def calculate_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(4096):
            file_hash.update(chunk)
        return file_hash.hexdigest()
