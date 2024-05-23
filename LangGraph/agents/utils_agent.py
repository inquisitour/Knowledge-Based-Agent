import os
import json
from dotenv import load_dotenv
from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

class UtilsAgent:
    def __init__(self, db_path):
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.graph = MessageGraph(memory=self.memory)
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("load_db_credentials", ToolNode(self.load_db_credentials))
        self.graph.add_node("save_db_credentials", ToolNode(self.save_db_credentials))
        self.graph.add_node("get_env_variable", ToolNode(self.get_env_variable))
        self.graph.set_entry_point("load_db_credentials")

    def load_db_credentials(self, db_type):
        load_dotenv()
        if db_type == 'postgres':
            return {
                'host': os.getenv('POSTGRES_HOST'),
                'port': os.getenv('POSTGRES_PORT'),
                'database': os.getenv('POSTGRES_DB'),
                'user': os.getenv('POSTGRES_USER'),
                'password': os.getenv('POSTGRES_PASSWORD')
            }
        elif db_type == 'neo4j':
            return {
                'uri': os.getenv('NEO4J_URI'),
                'username': os.getenv('NEO4J_USERNAME'),
                'password': os.getenv('NEO4J_PASSWORD')
            }
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def save_db_credentials(self, db_type, credentials):
        if db_type == 'postgres':
            with open('postgres_credentials.json', 'w') as f:
                json.dump(credentials, f)
        elif db_type == 'neo4j':
            with open('neo4j_credentials.json', 'w') as f:
                json.dump(credentials, f)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def get_env_variable(self, variable_name):
        value = os.getenv(variable_name)
        if value is None:
            raise ValueError(f"{variable_name} environment variable is not set.")
        return value

    def get_graph(self):
        return self.graph
def get_env_variable( variable_name):
        value = os.getenv(variable_name)
        if value is None:
            raise ValueError(f"{variable_name} environment variable is not set.")
        return value