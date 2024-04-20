import os
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from data_preprocessing import DBops  # Ensure DBops is properly defined and implemented

# Assuming OPENAI_API_KEY is set in the environment for security
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_default_openai_api_key")

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
    db_connection = DBops.get_database_connection()  # Ensure this method is properly implemented in DBops
    retriever = EmbeddingRetriever(embeddings, db_connection)
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')

    def retrieval_tool(query):
        documents = OpenAIops.retriever.retrieve(query)
        return "\n".join([f"Question: {doc['content']}\nAnswer: {doc['content']}" for doc in documents])

    tools = [
        Tool("retrieve", retrieval_tool, description="Retrieves similar questions based on embeddings.")
    ]

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    @staticmethod
    def answer_question(user_question):
        return OpenAIops.agent.run(user_question)

class ResponseAgent:
    def __init__(self):
        self.openaiops = OpenAIops()

    def answer_question(self, user_question):
        return self.openaiops.answer_question(user_question)

# Example usage
if __name__ == "__main__":
    response_agent = ResponseAgent()
    question = "How does the retrieval process work?"
    print(response_agent.answer_question(question))
