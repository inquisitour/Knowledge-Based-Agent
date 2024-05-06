import os
import json
import warnings
import numpy as np
import pandas as pd
from langchain.schema import HumanMessage
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

warnings.filterwarnings('ignore')

# Set environment variables
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "gravitas@123"

# Initialize the Neo4j Graph connection
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))

# Initialize models
llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')
embedding_model = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-large")

def batch_embeddings(texts):
    if not texts:
        return []
    embeddings = embedding_model.embed_documents(texts)
    return embeddings

def cosine_similarity(embedding1, embedding2):
    # Calculate cosine similarity between two embeddings
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def update_embeddings_in_graph(nodes_with_embeddings):
    query = """
        UNWIND $nodes as node
        MATCH (n)
        WHERE id(n) = node.id
        SET n.embedding = node.embedding
    """
    graph.query(query, params={"nodes": nodes_with_embeddings})

def create_knowledge_graph(csv_data):
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
        graph.query(query, params={
            "index": index,
            "question": question,
            "answer": answer,
            "category": category
        })

    # Fetch nodes and their labels for embedding
    all_nodes = graph.query("MATCH (n) RETURN id(n) as id, n.text as text")
    texts = [node['text'] for node in all_nodes if node['text']]
    embeddings = batch_embeddings(texts)
    nodes_with_embeddings = [{'id': node['id'], 'embedding': emb} for node, emb in zip(all_nodes, embeddings) if emb is not None]
    update_embeddings_in_graph(nodes_with_embeddings)

def query_knowledge_graph(user_query):
    prompt = f"Given the user query: {user_query}, generate a Cypher query to retrieve relevant information from the Neo4j knowledge graph."
    messages = HumanMessage(content=prompt)
    response = llm([messages])
    cypher_query = response.content

    results_dict = {}
    candidate_results = graph.query(cypher_query)
    if not candidate_results:
        user_query_embedding = embedding_model.embed_query(user_query)
        all_nodes = graph.query("MATCH (n) RETURN n.text as text, n.embedding as embedding")
        for node in all_nodes:
            if node['embedding'] is not None:
                score = cosine_similarity(user_query_embedding, node['embedding'])
                if score > 0.2:
                    results_dict[node['text']] = score
    else:
        user_query_embedding = embedding_model.embed_query(user_query)
        for node in candidate_results:
            if node['embedding'] is not None:
                score = cosine_similarity(user_query_embedding, node['embedding'])
                if score > 0.2:
                    results_dict[node['text']] = score

    return results_dict

def output_parser(user_query, results):
    # Open or create the JSON file and load existing data if any
    try:
        with open('query_results.json', 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Update the data dictionary with the new results
    data[user_query] = results

    # Write updated data back to the JSON file
    with open('query_results.json', 'w') as file:
        json.dump(data, file, indent=4)

# Load data from CSV
csv_data = pd.read_csv('categorized_qa_pairs.csv')
# Create knowledge graph
create_knowledge_graph(csv_data)

# Example usage
while True:
    user_query = input("Please enter your question or type 'exit' to quit: ")
    if user_query.lower() == 'exit':
        break
    results = query_knowledge_graph(user_query)
    if results:
        print("Query Results:", results)
        output_parser(user_query, results)
    else:
        print("No results found.")

