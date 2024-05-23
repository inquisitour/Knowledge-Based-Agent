import numpy as np
from typing import List
from langchain.schema import Document
from langgraph.graph import MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

class EmbeddingRetriever:
    def __init__(self, db_connection, db_path, embeddings):
        self.db_connection = db_connection
        self.embeddings = embeddings
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        self.graph = MessageGraph(memory=self.memory)
        self._setup_graph()
        print("Embedding retriever initialized")

    def _setup_graph(self):
        self.graph.add_node("retrieve_similar_questions", ToolNode(self.retrieve_similar_questions))
        self.graph.add_node("get_relevant_documents", ToolNode(self.get_relevant_documents))
        self.graph.set_entry_point("retrieve_similar_questions")

    def retrieve_similar_questions(self, query, k=20, min_similarity=0.1):
        query_vec = self.embeddings.embed_documents(query)[0]
        query_vec = np.array(query_vec)  # Ensure the query vector is writable
        query_vec /= np.linalg.norm(query_vec)
        similar_questions = []
        with self.db_connection.cursor() as cursor:
            cursor.execute("SELECT question, answer, embedding FROM faq_embeddings")
            results = cursor.fetchall()
            for result in results:
                question, answer, embedding = result
                embedding = np.frombuffer(embedding, dtype=np.float32).copy()  # Make a writable copy of the embedding
                embedding /= np.linalg.norm(embedding)
                similarity = np.dot(embedding, query_vec)
                if similarity >= min_similarity:
                    similar_questions.append({'question': question, 'answer': answer, 'similarity': similarity})
        similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_questions[:k]

    def get_relevant_documents(self, query: str) -> List[Document]:
        similar_questions = self.retrieve_similar_questions(query)
        documents = [Document(page_content=q['answer'], metadata={"question": q['question'], "similarity": q['similarity']}) for q in similar_questions]
        return documents

    def get_graph(self):
        return self.graph

# Example usage
if __name__ == "__main__":
    import psycopg2
    from agents.embedding_agent import EmbeddingAgent
    import os
    db_path = "embedding_retriever_memory.db"
    db_connection = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )
    embedding_agent = EmbeddingAgent(db_path=db_path)
    retriever = EmbeddingRetriever(db_connection=db_connection, db_path=db_path, embeddings=embedding_agent.embeddings)
    query = "What is artificial intelligence?"
    relevant_docs = retriever.get_relevant_documents(query)
    for doc in relevant_docs:
        print(doc)