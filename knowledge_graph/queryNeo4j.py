import os
import numpy as np
from langchain.schema import HumanMessage
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOpenAI

# Set environment variables
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "gravitas@123"

# Initialize the Neo4j Graph connection
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))

# Initialize the LangChain for Neo4j
llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')
embedding_model = 'text-embedding-3-large'  # Use the text-embedding-3-large model from OpenAI

def get_node_embedding(node_text):
    # Use LangChain to generate embeddings for the node text
    response = llm(f"Generate embeddings for: {node_text}", model=embedding_model)
    # Extract the embedding vector from the response
    embedding = response.choices[0].message['content']
    return embedding

def cosine_similarity(embedding1, embedding2):
    # Calculate cosine similarity between two embeddings
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def query_knowledge_graph(user_query):
    # Use LangChain to process the user query and generate a preliminary Cypher query
    prompt = f"Given the user query: {user_query}, generate a Cypher query to retrieve relevant information from the Neo4j knowledge graph."
    messages = HumanMessage(content=prompt)
    response = llm([messages])
    cypher_query = response.content
        
    # Execute the preliminary Cypher query to retrieve candidate nodes or paths
    candidate_results = graph.query(cypher_query)
    
    # If no results are fetched, return a message
    if not candidate_results:
        return "No results found."
    
    # Apply graph embeddings to the candidate nodes or paths and rank them based on similarity to the user query
    user_query_embedding = get_node_embedding(user_query)
    ranked_results = []
    for node in candidate_results:
        node_text = node['text']  # Assuming 'text' field contains the text of the node
        node_embedding = get_node_embedding(node_text)
        similarity_score = cosine_similarity(user_query_embedding, node_embedding)
        ranked_results.append((node, similarity_score))
    
    # Sort the results based on similarity scores in descending order
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top-ranked nodes or paths as the answer
    return [result[0] for result in ranked_results]

# Example usage
user_query = input("Please enter your question: ")
result = query_knowledge_graph(user_query)
if result:
    print("Query Results:", result)
else:
    print("No results found.")
