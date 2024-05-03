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

def get_node_embedding(node_text):
    if node_text is None:
        return None
    
    # Use OpenAIEmbeddings to generate embeddings for the node text
    embedding = embedding_model.embed_query(node_text)
    return embedding

def cosine_similarity(embedding1, embedding2):
    # Calculate cosine similarity between two embeddings
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def add_node_embedding(label, node_text, node_embedding):
    if node_embedding is None:
        return

    query = f"""
        MERGE (n:{label} {{text: $node_text}})
        SET n.embedding = $node_embedding
    """
    print(f"Executing query: {query}")
    print(f"With parameters: node_text={node_text}, node_embedding={node_embedding}")

    graph.query(query, params={
        "node_text": node_text,
        "node_embedding": node_embedding
    })

def create_knowledge_graph(csv_data):
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

    # Add embeddings to the nodes
    all_nodes = graph.query("MATCH (n) RETURN n.text, labels(n)")
    for node in all_nodes:
        node_text = node['n.text']
        node_labels = list(node['labels(n)'])
        node_embedding = get_node_embedding(node_text)

        if node_embedding is not None:
            if 'Question' in node_labels:
                add_node_embedding('Question', node_text, node_embedding)
            elif 'Answer' in node_labels:
                add_node_embedding('Answer', node_text, node_embedding)
            elif 'Category' in node_labels:
                add_node_embedding('Category', node_text, node_embedding)

def query_knowledge_graph(user_query):
    # Use LangChain to process the user query and generate a preliminary Cypher query
    prompt = f"Given the user query: {user_query}, generate a Cypher query to retrieve relevant information from the Neo4j knowledge graph."
    messages = HumanMessage(content=prompt)
    response = llm([messages])
    cypher_query = response.content

    # Check if candidate_results is empty
    candidate_results = graph.query(cypher_query)
    if not candidate_results:
        # If candidate_results is empty, perform graph vector similarity operation directly
        user_query_embedding = get_node_embedding(user_query)
        if user_query_embedding is not None:
            all_nodes = graph.query("MATCH (n) RETURN n.text")
            ranked_results = []
            for node in all_nodes:
                node_text = node['n.text']
                node_embedding = get_node_embedding(node_text)
                if node_embedding is not None:
                    similarity_score = cosine_similarity(user_query_embedding, node_embedding)
                    ranked_results.append((node_text, similarity_score))

            # Sort the results based on similarity scores in descending order
            ranked_results.sort(key=lambda x: x[1], reverse=True)

            # Return the top-ranked nodes as the answer
            return [result[0] for result in ranked_results]
        else:
            return []
    else:
        # If candidate_results is not empty, follow the original flow
        user_query_embedding = get_node_embedding(user_query)
        ranked_results = []
        for node in candidate_results:
            node_text = node['text']
            node_embedding = get_node_embedding(node_text)
            if node_embedding is not None:
                similarity_score = cosine_similarity(user_query_embedding, node_embedding)
                ranked_results.append((node, similarity_score))

        # Sort the results based on similarity scores in descending order
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # Return the top-ranked nodes or paths as the answer
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