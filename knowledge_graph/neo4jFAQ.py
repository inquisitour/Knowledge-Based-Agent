import os
import numpy as np
import pandas as pd
from langchain.schema import HumanMessage
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

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

    candidate_results = graph.query(cypher_query)
    if not candidate_results:
        user_query_embedding = embedding_model.embed_query(user_query)
        all_nodes = graph.query("MATCH (n) RETURN n.text as text, n.embedding as embedding")
        ranked_results = [(node['text'], cosine_similarity(user_query_embedding, node['embedding'])) for node in all_nodes if node['embedding'] is not None]
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in ranked_results]
    else:
        user_query_embedding = embedding_model.embed_query(user_query)
        ranked_results = [(node['text'], cosine_similarity(user_query_embedding, node['embedding'])) for node in candidate_results if node['embedding'] is not None]
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in ranked_results]

# Load data from CSV
csv_data = pd.read_csv('categorized_qa_pairs.csv')

# Create knowledge graph
create_knowledge_graph(csv_data)

# Example usage
user_query = input("Please enter your question: ")
result = query_knowledge_graph(user_query)
if result:
    print("Query Results:", result)
else:
    print("No results found.")
