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
        self.memory = SqliteSaver.from_conn_string(db_path)
        self.graph = MessageGraph()
        self._setup_graph()
        print("Embedding retriever initialized")

    def _setup_graph(self):
        self.graph.add_node("retrieve_similar_questions", ToolNode([self.retrieve_similar_questions]))
        self.graph.add_node("get_relevant_documents", ToolNode([self.get_relevant_documents]))

        self.graph.add_edge("retrieve_similar_questions", "get_relevant_documents")

        self.graph.set_entry_point("retrieve_similar_questions")

    def retrieve_similar_questions(self, query, k=20, min_similarity=0.1):
        """_summary_

        Args:
            query (_type_): _description_
            k (int, optional): _description_. Defaults to 20.
            min_similarity (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
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

                # Reshape the embeddings to match dimensionality
                if len(embedding) < len(query_vec):
                    embedding = np.pad(embedding, (0, len(query_vec) - len(embedding)), mode='constant')
                elif len(embedding) > len(query_vec):
                    query_vec = np.pad(query_vec, (0, len(embedding) - len(query_vec)), mode='constant')

                similarity = np.dot(embedding, query_vec)
                if similarity >= min_similarity:
                    similar_questions.append({'question': question, 'answer': answer, 'similarity': similarity})
        similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_questions[:k]

    def get_relevant_documents(self, query: str) -> List[Document]:
        """_summary_

        Args:
            query (str): _description_

        Returns:
            List[Document]: _description_
        """
        similar_questions = self.retrieve_similar_questions(query)
        documents = [Document(page_content=q['answer'], metadata={"question": q['question'], "similarity": q['similarity']}) for q in similar_questions]
        return documents
