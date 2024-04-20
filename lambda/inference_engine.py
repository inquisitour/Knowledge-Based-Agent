import os
import hashlib
import psycopg2
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from common import get_database_connection

# Environment variables for database configuration
DATABASE_URL = get_database_url()

# Ensure API keys are read from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class EmbeddingRetriever:
    def __init__(self, embeddings, db_connection):
        self.embeddings = embeddings
        self.db = db_connection

    def retrieve(self, query, k=5):
        query_vec = self.embeddings.embed_documents([query])[0]
        with self.db.cursor() as cursor:
            cursor.execute("SELECT id, content, embedding FROM documents")
            documents = cursor.fetchall()
        embeddings = np.array([doc[2] for doc in documents])
        distances = np.linalg.norm(embeddings - query_vec, axis=1)
        top_indices = np.argsort(distances)[:k]
        return [{'content': documents[i][1], 'id': documents[i][0]} for i in top_indices]

class OpenAIops:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    db_connection = psycopg2.connect(DATABASE_URL)
    retriever = EmbeddingRetriever(embeddings, db_connection)
    llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')

    def retrieval_tool(query):
        return "\n".join([f"Question: {doc['content']}\nAnswer: {doc['content']}" for doc in OpenAIops.retriever.retrieve(query)])

    tools = [
        Tool("retrieve", retrieval_tool, description="Retrieves documents based on query embeddings and returns them formatted as question-answer pairs.")
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    @staticmethod
    def answer_question(user_question):
        response = OpenAIops.agent.run(user_question)
        return response

class ResponseAgent:
    def __init__(self):
        self.openaiops = OpenAIops()

    def answer_question(self, user_question):
        return self.openaiops.answer_question(user_question)

OpenAIops.db_connection.close()  # Ensure you close the database connection when done
