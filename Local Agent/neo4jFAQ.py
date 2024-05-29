import os
import json
import faiss
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

class GraphEmbeddingRetriever(BaseModel):
    neo4j_uri: str = Field(os.getenv("NEO4J_URI"), description="URI for Neo4j database")
    neo4j_username: str = Field(os.getenv("NEO4J_USERNAME"), description="Username for Neo4j database")
    neo4j_password: str = Field(os.getenv("NEO4J_PASSWORD"), description="Password for Neo4j database")
    openai_api_key: str = Field(os.getenv("OPENAI_API_KEY"), description="OpenAI API key for embedding model")
    graph: Any = Field(None, description="Neo4jGraph instance")
    llm: Any = Field(None, description="Language model instance")  
    embedding_model: Any = Field(None, description="Embedding model instance")
    index: Any = Field(None, description="FAISS index instance")
    node_id_to_index: dict = Field(None, description="Mapping of node IDs to FAISS index IDs")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        ## Set environment variables
        # os.environ["NEO4J_URI"] = self.neo4j_uri
        # os.environ["NEO4J_USERNAME"] = self.neo4j_username
        # os.environ["NEO4J_PASSWORD"] = self.neo4j_password
        print(self.neo4j_uri)

        # Initialize the Neo4j Graph connection
        self.graph = Neo4jGraph(url=self.neo4j_uri, username=self.neo4j_username, password=self.neo4j_password)

        # Initialize models
        self.llm = ChatOpenAI(api_key=self.openai_api_key, model='gpt-3.5-turbo')
        self.embedding_model = OpenAIEmbeddings(api_key=self.openai_api_key, model="text-embedding-3-large")

        # Determine embedding dimension
        sample_embedding = self.embedding_model.embed_query("sample text")
        embedding_dim = len(sample_embedding)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.node_id_to_index = {}
        
        print("Graph embedding retriever initialized")

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
        #step 0: find embedding vector for the input user query
        user_query_embedding = self.embedding_model.embed_query(user_query)

        # Step 1: Generate candidate Cypher queries to find answer nodes closest to the user_query_embedding using cosine similarity



        # prompt = f"""Given the user query embedding ,
        #  generate a Cypher query [with no other text only the pure cypher query with the user_query_embedding varibale written inside curly brackets which can be directly be passed to execute no other comments] 
        #   to retrieve N relevant nodes from the Neo4j knowledge graph perform similarity search like cosine similarity or distance similarity. 
        #   The database credentials are: URI: {self.neo4j_uri}, Username: {self.neo4j_username}, Password: {self.neo4j_password}. 
        #   and the structure of the graph is 
        #   Nodes [ each node has a embedding property]
        #     Category (c)
        #         Properties:
        #         name: The name of the category.
        #     Question (q)
        #         Properties:
        #         id: Unique identifier for the question.
        #         text: The text of the question.
        #         category: The category to which the question belongs.
        #     Answer (a)
        #         Properties:
        #         id: Unique identifier for the answer (same as the question for direct association).
        #         text: The text of the answer.
        #         category: The category to which the answer belongs.
        #     Relationships
        #     HAS_ANSWER
        #         From: Question (q)
        #         To: Answer (a)
        #         Description: This relationship indicates that a particular question has a specific answer.
        #     INCLUDES
        #         From: Category (c)
        #         To: Question (q)
        #         Description: This relationship indicates that a category includes a particular question.
        #     From: Category (c)
        #         To: Answer (a)
        #         Description: This relationship indicates that a category includes a particular answer.
        #     """
        # print(prompt)
        # messages = HumanMessage(content=prompt)
        # response = self.llm([messages])
        # cypher_query = response.content

        # cypher_query = """
        # MATCH (q:Question)
        # WITH q, gds.alpha.similarity.cosine(q.embedding, {user_query_embedding}) AS similarity
        # ORDER BY similarity DESC
        # LIMIT 5
        # MATCH (q)-[:HAS_ANSWER]->(a:Answer)
        # RETURN q.text AS question, a.text AS answer, similarity

        # """
        cypher_query = """
        MATCH (n)
        RETURN n
        LIMIT 3
        """
        print("------the cypher query-------\n\n")
        # print(cypher_query)

        # Step 2: Execute candidate Cypher queries
        results_list = []
        seen_texts = set()  # Set to track texts and avoid duplicates
        candidate_results = self.graph.query(cypher_query.format(user_query_embedding=user_query_embedding))
        print("------the candidate results-------\n\n")
        print(candidate_results)
        print("\n-----xox----\n")

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
        print("\n-----holo----\n")
        print(results_list)
        return results_list

    # def query_knowledge_graph(self, user_query: str) -> List[Dict[str, any]]:
    #     """
    #     Query the Neo4j knowledge graph based on the user's input and return relevant results.

    #     Args:
    #         user_query (str): User's query.

    #     Returns:
    #         List[Dict[str, any]]: List of relevant results with information like text, score, label, and category.
    #     """
    #     # Step 1: Generate candidate Cypher queries
    #     prompt = f"Given the user query: {user_query}, generate a Cypher query to retrieve relevant information from the Neo4j knowledge graph."
    #     messages = HumanMessage(content=prompt)
    #     response = self.llm([messages])
    #     cypher_query = response.content
    #     print("-------------\n\n")
    #     print(cypher_query)
    #     # Step 2: Execute candidate Cypher queries
    #     results_list = []
    #     seen_texts = set()  # Set to track texts and avoid duplicates
    #     try:
    #         candidate_results = self.graph.query(cypher_query)
    #     except Exception as e:
    #         print(f"Error executing Cypher query: {e}")
    #         candidate_results = []

    #     if candidate_results:
    #         for node in candidate_results:
    #             if node['text'] not in seen_texts:
    #                 results_list.append({
    #                     'text': node['text'],
    #                     'score': 0.2,  # High score for direct matches
    #                     'label': node['labels'][0] if node['labels'] else 'No Label',
    #                     'category': node['category'] if 'category' in node else 'No Category'
    #                 })
    #                 seen_texts.add(node['text'])

    #     # Step 3: FAISS-based embedding similarity search
    #     user_query_embedding = self.embedding_model.embed_query(user_query)
    #     D, I = self.index.search(np.array([user_query_embedding], dtype=np.float32), k=10)  # Retrieve top 10 matches

    #     for i in range(len(I[0])):
    #         node_index = I[0][i]
    #         if node_index >= 0 and node_index in self.node_id_to_index:  # Check if node_index is valid
    #             node_id = self.node_id_to_index[node_index]
    #             score = 1 - D[0][i]  # Convert distance to similarity score
    #             node_data = self.graph.query(f"MATCH (n) WHERE id(n) = {node_id} OPTIONAL MATCH (n)<-[:INCLUDES]-(c:Category) RETURN n.text as text, labels(n) as labels, coalesce(c.name, 'No Category') as category")[0]
    #             if node_data['text'] not in seen_texts:
    #                 results_list.append({
    #                     'text': node_data['text'],
    #                     'score': score,
    #                     'label': node_data['labels'][0] if node_data['labels'] else 'No Label',
    #                     'category': node_data['category']
    #                 })
    #                 seen_texts.add(node_data['text'])  # Add text to set to track as seen

    #     return results_list

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
