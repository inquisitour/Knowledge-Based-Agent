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
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
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
    db: Any = Field(None, description="Neo4jVector instance")
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

    def load_knowledge_graph(self) -> None:
        self.db = Neo4jVector.from_existing_index(OpenAIEmbeddings() ,url=self.neo4j_uri, username=self.neo4j_username, password=self.neo4j_password,index_name="vector",)

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
            question_id = f"q_{index}"  # Unique identifier for each question
            query = """
                MERGE (c:Category {name: $category})
                MERGE (q:Question {question_id: $question_id, text: $question, category: $category})
                MERGE (a:Answer {question_id: $question_id, text: $answer, category: $category})
                MERGE (q)-[:HAS_ANSWER]->(a)
                MERGE (c)-[:INCLUDES]->(q)
                MERGE (c)-[:INCLUDES]->(a)
            """
            self.graph.query(query, params={
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "category": category
            })

        self.db = Neo4jVector.from_existing_graph(
            embedding=OpenAIEmbeddings(),
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name="vector",
            node_label="Question",
            text_node_properties=["text", "category"],
            embedding_node_property="embedding",
        )

       

    

    def query_knowledge_graph(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Query the Neo4j knowledge graph based on the user's input and return relevant results.

        Args:
            user_query (str): User's query.

        Returns:
            List[Dict[str, Any]]: List of relevant results with question and answer texts.
        """
        # Perform similarity search to get the most relevant questions
        docs_with_score = self.db.similarity_search_with_score(user_query, k=5)
        
        # Collect question_ids of the nodes with high similarity
        for i,_ in docs_with_score:
            print(i.metadata)
        question_ids = [doc.metadata['question_id'] for doc, _ in docs_with_score]

        # Fetch questions and their associated answers using question_ids
        query = """
        MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)
        WHERE q.question_id IN $question_ids
        RETURN q, collect(a) AS answers
        """
        
        result = self.graph.query(query, params={"question_ids": question_ids})
        
        questions_with_answers = []
        for record in result:
            question_node = record["q"]
            answer_nodes = record["answers"]
            question_with_answers = {
                "question": question_node["text"],
                "answers": [answer["text"] for answer in answer_nodes]
            }
            questions_with_answers.append(question_with_answers)
        results_list = []
        for qa in questions_with_answers:
            # print("-" * 80)
            # print("Question: ", qa["question"])
            for answer in qa["answers"]:
                # print("Answer: ", answer)
                # print("-" * 80)
                results_list.append({
                        'question':  qa["question"],
                        'answer': answer,
                    })
        
        return results_list
        



# ret = GraphEmbeddingRetriever()
# # ret.create_knowledge_graph(pd.read_csv("C:/Users/Rudra/main/code/gravitas/Knowledge-Based-Agent/Local Agent/categorized_qa_pairs.csv", encoding='latin1'))
# ans = ret.query_knowledge_graph("what does my eye hurt")
# print(ans)