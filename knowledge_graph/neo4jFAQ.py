import os
import pandas as pd
from langchain_community.graphs import Neo4jGraph

# Set environment variables
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "gravitas@123"

# Initialize the Neo4j Graph connection
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))

# Load the CSV file using pandas
csv_data = pd.read_csv('categorized_qa_pairs.csv')  

# Preprocess the data and create nodes and relationships in Neo4j using Cypher queries
for index, row in csv_data.iterrows():
    question = row['questions']
    answer = row['answers']
    category = row['category']  

    # Include the category in the node and create more detailed relationships
    create_query = """
    MERGE (c:Category {name: $category})
    MERGE (q:Question {id: $index, text: $question, category: $category})
    MERGE (a:Answer {id: $index, text: $answer, category: $category})
    MERGE (q)-[:HAS_ANSWER]->(a)
    MERGE (c)-[:INCLUDES]->(q)
    MERGE (c)-[:INCLUDES]->(a)
    """
    try:
        graph.query(create_query, {'index': index, 'question': question, 'answer': answer, 'category': category})
        print(f"Processed row {index + 1} in category '{category}'")
    except Exception as e:
        print(f"Failed to process row {index + 1} in category '{category}': {str(e)}")

