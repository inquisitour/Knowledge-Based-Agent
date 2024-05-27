from contextlib import contextmanager
import psycopg2
import hashlib
import pickle
import numpy as np
from functools import wraps
from psycopg2 import pool, extras
from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.utils_agent import UtilsAgent
from agents.embedding_agent import EmbeddingAgent


# Centralized connection management
class DBops:
    def __init__(self, db_path):
        self.db_config = UtilsAgent.load_db_credentials('postgres')
        self.embeddings = EmbeddingAgent(db_path)
        self.memory = SqliteSaver.from_conn_string(db_path)
        self.graph = MessageGraph()
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,  # Adjust maxconn based on your expected workload
            **self.db_config
        )
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("get_db_connection", ToolNode([self.get_database_connection]))
        self.graph.add_node("setup_database", ToolNode([self.setup_database]))
        self.graph.add_node("process_local_file", ToolNode([self.process_local_file]))
        self.graph.add_node("check_data_hash", ToolNode([self.check_data_hash]))
        self.graph.add_node("delete_all_data_hashes", ToolNode([self.delete_all_data_hashes]))
        self.graph.add_node("update_data_hash", ToolNode([self.update_data_hash]))

        self.graph.add_edge("get_db_connection", "setup_database")
        self.graph.add_edge("setup_database", "process_local_file")
        self.graph.add_edge("process_local_file", "check_data_hash")
        self.graph.add_conditional_edges("check_data_hash", self.condition_check)
        self.graph.add_edge("delete_all_data_hashes", "update_data_hash")

        self.graph.set_entry_point("get_db_connection")

    def condition_check(self, state):
        if not state["check_data_hash"]:
            return "delete_all_data_hashes"
        return "__end__"

    @contextmanager
    def get_database_connection(self):
        """Get a database connection from the connection pool."""
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    def with_connection(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.get_database_connection() as conn:
                return func(self, *args, **kwargs, conn=conn)
        return wrapper
    
    def calculate_file_hash(self, file_content):
        try:
            return hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            print(f"Error calculating file hash: {e}")

    def process_local_file(self, data_csv):
        """Process the local CSV file and update the database if necessary."""
        file_content = pickle.dumps(data_csv)
        file_hash = self.calculate_file_hash(file_content)
        if not self.check_data_hash(file_hash):
            print("Data hash mismatch found. Updating database...")
            
            csv_data = pickle.loads(file_content)
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
    def setup_database(self, conn):
        """
        Set up the necessary database tables.

        Args:
            conn (psycopg2.extensions.connection): The database connection.
        """
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

    @with_connection
    def check_data_hash(self, file_hash, conn):
        """Check if the given file hash exists in the database."""
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM data_hash WHERE file_hash = %s", (file_hash,))
            return cur.fetchone() is not None

    @with_connection
    def insert_data(self, questions, answers, embeddings, conn):
        """Insert the given data into the database."""
        print("Inserting data into database")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM faq_embeddings")
            args = [(q, a, psycopg2.Binary(np.array(emb).astype(np.float32).tobytes())) for q, a, emb in zip(questions, answers, embeddings)]
            extras.execute_batch(cur, "INSERT INTO faq_embeddings (question, answer, embedding) VALUES (%s, %s, %s)", args)
            conn.commit()

    @with_connection
    def delete_all_data_hashes(self, conn):
        """Delete all the data hashes from the database."""
        with conn.cursor() as cur:
            cur.execute("DELETE FROM data_hash")
            conn.commit()

    @with_connection
    def update_data_hash(self, file_hash, conn):
        """Update the database with the given file hash."""
        with conn.cursor() as cur:
            cur.execute("INSERT INTO data_hash (file_hash) VALUES (%s)", (file_hash,))
            conn.commit()