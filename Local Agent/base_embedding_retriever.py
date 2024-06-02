
import os
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from data_processing import get_database_connection
from typing import List, Any
from langchain.schema import Document
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from pydantic import BaseModel, Field

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

class EmbeddingRetriever(BaseModel):
    db_connection: Any = Field(..., description="Database connection for retrieving embeddings")
    embeddings: Any = Field(None, description="OpenAI embeddings model")

    def __init__(self, db_connection):
        super().__init__(db_connection=db_connection)
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")
        print("Embedding retriever initialized")

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
                #print(similarity)
                if similarity >= min_similarity:
                    similar_questions.append({'question': question, 'answer': answer, 'similarity': similarity})
            similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_questions[:k]
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        similar_questions = self.retrieve_similar_questions(query)
        documents = [Document(page_content=q['answer'], metadata={"question": q['question'], "similarity": q['similarity']}) for q in similar_questions]
        print("in normal retrivier ---\nRetrieved documents: ", documents)
        return documents