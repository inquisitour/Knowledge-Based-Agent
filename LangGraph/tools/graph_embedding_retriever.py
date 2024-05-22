from neo4j import GraphDatabase
from langchain_community.chat_models import ChatOpenAI
import numpy as np
import faiss
from agents.utils_agent import get_env_variable
from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage
import pandas as pd

class GraphEmbeddingRetriever:
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password, openai_api_key, db_path):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
        self.embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.graph_manager = MessageGraph(memory=self.memory)
        self._setup_graph()

        sample_embedding = self.embedding_model.embed_query("sample text")
        embedding_dim = len(sample_embedding)

        self.index = faiss.IndexFlatL2(embedding_dim)
        self.node_id_to_index = {}
        
        print("Graph embedding retriever initialized")

    def _setup_graph(self):
        self.graph_manager.add_node("create_knowledge_graph", ToolNode(self.create_knowledge_graph))
        self.graph_manager.add_node("query_knowledge_graph", ToolNode(self.query_knowledge_graph))
        self.graph_manager.set_entry_point("create_knowledge_graph")

    def batch_embeddings(self, texts):
        if not texts:
            return []
        embeddings = self.embedding_model.embed_documents(texts)
        return embeddings

    def create_knowledge_graph(self, csv_data):
        with self.driver.session() as session:
            for index, row in csv_data.iterrows():
                question = row['questions']
                answer = row['answers']
                category = row['category']
                query = """
                    MERGE (c:Category {name: $category})
                    MERGE (q:Question {id: $index, text: $question, category: $category})
                    MERGE (a:Answer {id: $index, text: $answer, category: $category})
                    MERGE (q)-[:HAS_ANSWER]->(a)
                    MERGE (c)-[:INCLUDES]->(q)
                    MERGE (c)-[:INCLUDES]->(a)
                """
                session.run(query, index=index, question=question, answer=answer, category=category)

            all_nodes = session.run("MATCH (n) RETURN id(n) as id, n.text as text").data()
            texts = [node['text'] for node in all_nodes if node['text']]
            embeddings = self.batch_embeddings(texts)
            nodes_with_embeddings = [{'id': node['id'], 'embedding': emb} for node, emb in zip(all_nodes, embeddings) if emb is not None]

            self.update_embeddings_in_graph(nodes_with_embeddings)
            self.build_faiss_index(nodes_with_embeddings)

    def update_embeddings_in_graph(self, nodes_with_embeddings):
        with self.driver.session() as session:
            query = """
                UNWIND $nodes as node
                MATCH (n)
                WHERE id(n) = node.id
                SET n.embedding = node.embedding
            """
            session.run(query, nodes=nodes_with_embeddings)

    def build_faiss_index(self, nodes_with_embeddings):
        embeddings = [node['embedding'] for node in nodes_with_embeddings]
        node_ids = [node['id'] for node in nodes_with_embeddings]
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.node_id_to_index = {idx: node_id for idx, node_id in enumerate(node_ids)}

    def query_knowledge_graph(self, user_query):
        cypher_query = self.generate_cypher_query(user_query)
        results_list = []
        seen_texts = set()  # Set to track texts and avoid duplicates
        try:
            with self.driver.session() as session:
                candidate_results = session.run(cypher_query).data()
        except Exception as e:
            print(f"Error executing Cypher query: {e}")
            candidate_results = []
        
        if candidate_results:
            for node in candidate_results:
                if node['text'] not in seen_texts:
                    results_list.append({
                        'text': node['text'],
                        'score': 0.2,  # High score for direct matches
                        'label': node['labels'][0] if node['labels'] else 'No Label',
                        'category': node['category'] if 'category' in node else 'No Category'
                    })
                    seen_texts.add(node['text'])

        user_query_embedding = self.embedding_model.embed_query(user_query)
        D, I = self.index.search(np.array([user_query_embedding], dtype=np.float32), k=10)  # Retrieve top 10 matches

        for i in range(len(I[0])):
            node_index = I[0][i]
            if node_index >= 0 and node_index in self.node_id_to_index:  # Check if node_index is valid
                node_id = self.node_id_to_index[node_index]
                score = 1 - D[0][i]  # Convert distance to similarity score
                with self.driver.session() as session:
                    node_data = session.run(f"MATCH (n) WHERE id(n) = {node_id} OPTIONAL MATCH (n)<-[:INCLUDES]-(c:Category) RETURN n.text as text, labels(n) as labels, coalesce(c.name, 'No Category') as category").data()[0]
                if node_data['text'] not in seen_texts:
                    results_list.append({
                        'text': node_data['text'],
                        'score': score,
                        'label': node_data['labels'][0] if node_data['labels'] else 'No Label',
                        'category': node_data['category']
                    })
                    seen_texts.add(node_data['text'])  # Add text to set to track as seen

        return results_list

    def generate_cypher_query(self, user_query):
        prompt = f"Given the user query: {user_query}, generate a Cypher query to retrieve relevant information from the Neo4j knowledge graph."
        messages = HumanMessage(content=prompt)
        response = self.llm([messages])
        cypher_query = response.content
        return cypher_query

    def process_create_knowledge_graph(self, csv_data):
        return self.graph_manager.run("create_knowledge_graph", csv_data=csv_data)

    def process_query_knowledge_graph(self, user_query):
        return self.graph_manager.run("query_knowledge_graph", user_query=user_query)

# Example usage
if __name__ == "__main__":
    neo4j_uri = get_env_variable("NEO4J_URI")
    neo4j_username = get_env_variable("NEO4J_USERNAME")
    neo4j_password = get_env_variable("NEO4J_PASSWORD")
    openai_api_key = get_env_variable("OPENAI_API_KEY")
    db_path = "graph_embedding_retriever_memory.db"

    retriever = GraphEmbeddingRetriever(neo4j_uri, neo4j_username, neo4j_password, openai_api_key, db_path)
    data_csv = pd.DataFrame({
        'questions': ["What is AI?", "What is machine learning?"],
        'answers': ["AI is artificial intelligence.", "Machine learning is a subset of AI."],
        'category': ["Technology", "Technology"]
    })
    retriever.process_create_knowledge_graph(data_csv)
    user_query = "Tell me about AI."
    results = retriever.process_query_knowledge_graph(user_query)
    print(results)
