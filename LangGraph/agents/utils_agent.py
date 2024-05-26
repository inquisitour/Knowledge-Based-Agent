import os
import sqlite3
from dotenv import load_dotenv
from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

class UtilsAgent:
    def __init__(self, db_path):
        try:
            self.memory = SqliteSaver.from_conn_string(db_path)
            self.graph = MessageGraph()
            self._setup_graph()
        except sqlite3.OperationalError as e:
            print(f"Error opening database file: {e}")

    def _setup_graph(self):
        self.graph.add_node("get_env_variable", ToolNode([UtilsAgent.get_env_variable]))
        self.graph.add_node("load_db_credentials", ToolNode([UtilsAgent.load_db_credentials]))

        self.graph.add_edge("get_env_variable", "load_db_credentials")

        self.graph.set_entry_point("get_env_variable")

    @staticmethod
    def get_env_variable(variable_name):
        """Get the value of an environment variable."""
        value = os.getenv(variable_name)
        if value is None:
            raise ValueError(f"{variable_name} environment variable is not set.")
        return value

    @staticmethod
    def load_db_credentials(db_type):
        """Load database credentials from environment variables."""
        load_dotenv()
        if db_type == 'postgres':
            return {
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT'),
                'database': os.getenv('DB_DB'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            }
        elif db_type == 'neo4j':
            return {
                'uri': os.getenv('NEO4J_URI'),
                'username': os.getenv('NEO4J_USERNAME'),
                'password': os.getenv('NEO4J_PASSWORD')
            }
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
