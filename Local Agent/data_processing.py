import os
import psycopg2
from psycopg2 import pool, extras
from hashlib import sha256
import pickle
import numpy as np
import pandas as pd
from contextlib import contextmanager
from langchain.embeddings.openai import OpenAIEmbeddings
import networkx as nx
from pyvis.network import Network 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

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
        try:
            return sha256(file_content).hexdigest()
        except Exception as e:
            print(f"Error calculating file hash: {e}")

    def process_local_file(self, excel_file_content, csv_file_content):
        try:
            excel_file_content = pickle.dumps(excel_file_content)
            excel_file_hash = self.calculate_file_hash(excel_file_content)

            csv_file_content = pickle.dumps(csv_file_content)
            csv_file_hash = self.calculate_file_hash(csv_file_content)
            
            if  not self.check_data_hash(excel_file_hash) or not self.check_data_hash(csv_file_hash):
                print("Data hash mismatch found. Updating database...")
                
                excel_data = pickle.loads(excel_file_content)
                excel_G = self.create_knowledge_graph(excel_data)
                print("Knowledge graph created for excel data, preparing graph embeddings")
                excel_embeddings = self.embed_graph_data(excel_G)

                csv_data = pickle.loads(csv_file_content)
                csv_G = self.create_knowledge_graph(csv_data)
                print("Knowledge graph created for csv data, preparing graph embeddings")
                csv_embeddings = self.embed_graph_data(csv_G)
                
                
                embeddings = {**excel_embeddings, **csv_embeddings}
                questions = list(excel_G.nodes()) + list(csv_G.nodes())
                answers = list(excel_G.nodes()) + list(csv_G.nodes())
                
                self.insert_data(questions, answers, embeddings)
                self.delete_all_data_hashes()
                self.update_data_hash(excel_file_hash)
                self.update_data_hash(csv_file_hash)
                print("Database updated with new data and data hash")
            else:
                print("Data is up to date")
        except Exception as e:
            print(f"Error processing local file: {e}")
    
    def create_knowledge_graph(self, data):
        try:
            G = nx.DiGraph()
            core_categories = ['Opthalmologist', 'Nutrition and Eye Health', 'Lifestyle Factors Affecting Eye', 'Screen Time and Eye Strain', 'Eye Exercises and Vision', 'Medication and Eye Care', 'Eye Disease Prevention Tips']
            
            def add_category_node(category):
                G.add_node(category, type='category', title=category)
                if category in core_categories:
                    G.nodes[category]['core'] = True  # Mark core categories

            if isinstance(data, pd.DataFrame):
                csv_data = {'General': data}  # Default category for csv data
                data = csv_data
            
            for category, df in data.items():
                category = category.strip()  # Trim any excess whitespace from the category name
                add_category_node(category)
                    
                # Add nodes and edges for questions and answers within the current category
                for _, row in df.iterrows():
                    question = row.get('questions')
                    answer = row.get('answers')
                    if pd.notna(question):
                        G.add_node(question, type='question', title=question)
                        G.add_edge(category, question, relationship='has_question')
                        if pd.notna(answer):
                            G.add_node(answer, type='answer', title=answer)
                            G.add_edge(question, answer, relationship='answered_by')
            return G
        except Exception as e:
            print(f"Error creating knowledge graph: {e}")
            return None
    
    def embed_graph_data(self, G):
        try:
            embeddings = {}
            questions = [node for node in G.nodes if G.nodes[node]['type'] in ['question', 'answer']]
            embedded_nodes = self.embeddings.embed_documents(questions)
            for node, embedding in zip(questions, embedded_nodes):
                embeddings[node] = embedding
                G.nodes[node]['embedding'] = embedding
            print("Graph embeddings created successfully")
            return embeddings
        except Exception as e:
            print(f"Error embedding graph data: {e}")
            return {}

    @with_connection
    def insert_data(self, questions, answers, embeddings, conn):
        try:
            print("Inserting data into database")
            with conn.cursor() as cur:
                cur.execute("DELETE FROM faq_embeddings")
                args = [(q, a, psycopg2.Binary(np.array(emb).tobytes())) for q, a, emb in zip(questions, answers, embeddings)]
                extras.execute_batch(cur, "INSERT INTO faq_embeddings (question, answer, embedding) VALUES (%s, %s, %s)", args)
                print("Data inserted into database successfully")
                conn.commit()
        except Exception as e:
            print(f"Error inserting data into database: {e}")

    @with_connection
    def check_data_hash(self, file_hash, conn):
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS(SELECT 1 FROM data_hash WHERE file_hash = %s)", (file_hash,))
                print("Checking data hash in database")
                return cur.fetchone()[0]
        except Exception as e:
            print(f"Error checking data hash: {e}")
            return False

    @with_connection
    def update_data_hash(self, file_hash, conn):
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO data_hash (file_hash) VALUES (%s) ON CONFLICT (file_hash) DO NOTHING", (file_hash,))
                print("Updating data hash in database")
                conn.commit()
        except Exception as e:
            print(f"Error updating data hash: {e}")

    @with_connection
    def delete_all_data_hashes(self, conn):
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM data_hash")
                conn.commit()
                print("Deleted all data hashes from the database.")
        except Exception as e:
            print(f"Error deleting all data hashes: {e}")

    @with_connection
    def setup_database(self, conn):
        try:
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
        except Exception as e:
            print(f"Error setting up database: {e}")

    def visualize_graph(self, *graphs):
        try:
            combined_graph = nx.compose_all(graphs)
            nt = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
            nt.from_nx(combined_graph)
            print("Knowledge graph is ready for visualization")
            print("Visualizing knowledge graph")
            nt.show("knowledge_graph.html", notebook=False)
        except Exception as e:
            print(f"Error visualizing graph: {e}")

# Uncomment to visualize knowledge graph            
# db_ops.visualize_graph(db_ops.create_knowledge_graph(excel_file_content), db_ops.create_knowledge_graph(csv_file_content))

