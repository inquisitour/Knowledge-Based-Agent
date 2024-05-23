from contextlib import contextmanager
import psycopg2
import hashlib
import pandas as pd
import pickle
import numpy as np
from psycopg2 import pool, extras
from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.utils_agent import load_db_credentials

# Centralized connection management
class DBops:
    def __init__(self, db_path, embeddings):
        self.db_config = load_db_credentials('postgres')
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.graph = MessageGraph(memory=self.memory)
        self.embeddings = embeddings
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("process_local_file", ToolNode(self.process_local_file))
        self.graph.add_node("setup_database", ToolNode(self.setup_database))
        self.graph.add_node("check_data_hash", ToolNode(self.check_data_hash))
        self.graph.add_node("delete_all_data_hashes", ToolNode(self.delete_all_data_hashes))
        self.graph.add_node("update_data_hash", ToolNode(self.update_data_hash))
        self.graph.set_entry_point("setup_database")

    @contextmanager
    def get_database_connection(self):
        conn = psycopg2.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()

    def with_connection(func):
        def wrapper(self, *args, **kwargs):
            with self.get_database_connection() as conn:
                return func(self, *args, **kwargs, conn=conn)
        return wrapper

    @with_connection
    def process_local_file(self, data_csv, conn):
        file_content = pickle.dumps(data_csv)
        file_hash = self.calculate_file_hash(file_content)
        if not self.check_data_hash(file_hash, conn):
            print("Data hash mismatch found. Updating database...")
            
            csv_data = pickle.loads(file_content)
            if 'questions' in csv_data.columns and 'answers' in csv_data.columns:
                questions = csv_data['questions'].tolist()
                answers = csv_data['answers'].tolist()
                embeddings = self.embeddings.embed_documents(questions)  # Batch processing
                self.insert_data(questions, answers, embeddings, conn)
                self.delete_all_data_hashes(conn)
                self.update_data_hash(file_hash, conn)
                print("Database updated with new data and data hash")
            else:
                raise ValueError("CSV does not contain the required 'questions' and 'answers' columns")
        else:
            print("Data is up to date")

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

    def calculate_file_hash(self, file_content):
        return hashlib.sha256(file_content).hexdigest()

    @with_connection
    def check_data_hash(self, file_hash, conn):
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM data_hash WHERE file_hash = %s", (file_hash,))
            return cur.fetchone() is not None

    @with_connection
    def insert_data(self, questions, answers, embeddings, conn):
        print("Inserting data into database")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM faq_embeddings")
            args = [(q, a, psycopg2.Binary(np.array(emb).astype(np.float32).tobytes())) for q, a, emb in zip(questions, answers, embeddings)]
            extras.execute_batch(cur, "INSERT INTO faq_embeddings (question, answer, embedding) VALUES (%s, %s, %s)", args)
            conn.commit()

    @with_connection
    def delete_all_data_hashes(self, conn):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM data_hash")
            conn.commit()

    @with_connection
    def update_data_hash(self, file_hash, conn):
        with conn.cursor() as cur:
            cur.execute("INSERT INTO data_hash (file_hash) VALUES (%s)", (file_hash,))
            conn.commit()

    def get_graph(self):
        return self.graph