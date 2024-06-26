import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict
from langchain.schema import HumanMessage
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import faiss

warnings.filterwarnings('ignore')

class GraphEmbeddingRetriever:
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password, openai_api_key):
        # Set environment variables
        os.environ["NEO4J_URI"] = neo4j_uri
        os.environ["NEO4J_USERNAME"] = neo4j_username
        os.environ["NEO4J_PASSWORD"] = neo4j_password

        # Initialize the Neo4j Graph connection
        self.graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))

        # Initialize models
        self.llm = ChatOpenAI(api_key=openai_api_key, model='gpt-3.5-turbo')
        self.embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large")

        # Determine embedding dimension
        sample_embedding = self.embedding_model.embed_query("sample text")
        embedding_dim = len(sample_embedding)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.node_id_to_index = {}

    def batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Calculate embeddings for a list of texts using the OpenAI embedding model.

        Args:
            texts (List[str]): List of texts to calculate embeddings for.

        Returns:
            List[List[float]]: List of embeddings for the input texts.
        """
        if not texts:
            return []
        embeddings = self.embedding_model.embed_documents(texts)
        return embeddings

    def create_knowledge_graph(self, csv_data: pd.DataFrame) -> None:
        """
        Create a knowledge graph in Neo4j based on the input CSV data and build a vector index.

        Args:
            csv_data (pd.DataFrame): DataFrame containing the data to be used for creating the knowledge graph.
        """
        # Create nodes and relationships in the graph
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
            self.graph.query(query, params={
                "index": index,
                "question": question,
                "answer": answer,
                "category": category
            })

        # Fetch nodes and their labels for embedding
        all_nodes = self.graph.query("MATCH (n) RETURN id(n) as id, n.text as text")
        texts = [node['text'] for node in all_nodes if node['text']]
        embeddings = self.batch_embeddings(texts)
        nodes_with_embeddings = [{'id': node['id'], 'embedding': emb} for node, emb in zip(all_nodes, embeddings) if emb is not None]

        # Update embeddings in graph and build FAISS index
        self.update_embeddings_in_graph(nodes_with_embeddings)
        self.build_faiss_index(nodes_with_embeddings)

    def update_embeddings_in_graph(self, nodes_with_embeddings: List[Dict[str, any]]) -> None:
        """
        Update the embeddings of nodes in the Neo4j graph.

        Args:
            nodes_with_embeddings (List[Dict[str, any]]): List of node dictionaries containing node IDs and embeddings.
        """
        query = """
            UNWIND $nodes as node
            MATCH (n)
            WHERE id(n) = node.id
            SET n.embedding = node.embedding
        """
        self.graph.query(query, params={"nodes": nodes_with_embeddings})

    def build_faiss_index(self, nodes_with_embeddings: List[Dict[str, any]]) -> None:
        """
        Build a FAISS vector index from the embeddings.

        Args:
            nodes_with_embeddings (List[Dict[str, any]]): List of node dictionaries containing node IDs and embeddings.
        """
        embeddings = [node['embedding'] for node in nodes_with_embeddings]
        node_ids = [node['id'] for node in nodes_with_embeddings]
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.node_id_to_index = {idx: node_id for idx, node_id in enumerate(node_ids)}

    def query_knowledge_graph(self, user_query: str) -> List[Dict[str, any]]:
        """
        Query the Neo4j knowledge graph based on the user's input and return relevant results.

        Args:
            user_query (str): User's query.

        Returns:
            List[Dict[str, any]]: List of relevant results with information like text, score, label, and category.
        """
        # Step 1: Generate candidate Cypher queries
        prompt = f"Given the user query: {user_query}, generate a Cypher query to retrieve relevant information from the Neo4j knowledge graph."
        messages = HumanMessage(content=prompt)
        response = self.llm([messages])
        cypher_query = response.content

        # Step 2: Execute candidate Cypher queries
        results_list = []
        seen_texts = set()  # Set to track texts and avoid duplicates
        candidate_results = self.graph.query(cypher_query)
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

        # Step 3: FAISS-based embedding similarity search
        user_query_embedding = self.embedding_model.embed_query(user_query)
        D, I = self.index.search(np.array([user_query_embedding], dtype=np.float32), k=10)  # Retrieve top 10 matches

        for i in range(len(I[0])):
            node_id = self.node_id_to_index[I[0][i]]
            score = 1 - D[0][i]  # Convert distance to similarity score
            node_data = self.graph.query(f"MATCH (n) WHERE id(n) = {node_id} OPTIONAL MATCH (n)<-[:INCLUDES]-(c:Category) RETURN n.text as text, labels(n) as labels, coalesce(c.name, 'No Category') as category")[0]
            if node_data['text'] not in seen_texts:
                results_list.append({
                    'text': node_data['text'],
                    'score': score,
                    'label': node_data['labels'][0] if node_data['labels'] else 'No Label',
                    'category': node_data['category']
                })
                seen_texts.add(node_data['text'])  # Add text to set to track as seen

        return results_list

    def output_parser(self, user_query: str, results: List[Dict[str, any]]) -> None:
        """
        Write the query results to a JSON file.

        Args:
            user_query (str): User's query.
            results (List[Dict[str, any]]): List of relevant results.
        """
        try:
            with open('query_results.json', 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        data[user_query] = results

        with open('query_results.json', 'w') as file:
            json.dump(data, file, indent=4)

# Example usage
if __name__ == "__main__":
    # Initialize the GraphEmbeddingRetriever instance
    retriever = GraphEmbeddingRetriever(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="gravitas@123",
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

    # Load data from CSV
    csv_data = pd.read_csv('categorized_qa_pairs.csv')

    # Create knowledge graph
    retriever.create_knowledge_graph(csv_data)

    # Run the query loop
    while True:
        user_query = input("Please enter your question or type 'exit' to quit: ")
        if user_query.lower() == 'exit':
            break
        results = retriever.query_knowledge_graph(user_query)
        if results:
            print("Query Results:", results)
            retriever.output_parser(user_query, results)
        else:
            print("No results found.")
