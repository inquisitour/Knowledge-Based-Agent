import os
import hashlib
import psycopg2
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from data_preprocessing import DBops, get_database_url

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

    PREFIX = (
        "Answer the question as detailed as possible from the provided context which can be info from "
        "the questions or the answers, make sure to provide all the details, account for spelling errors assume "
        "the closest meaning of the question, if information about the question or any similar question is not in provided context "
        "just say, 'answer is not available in the context', don't provide the wrong answer\n\n"
    )

    FORMAT_INSTRUCTIONS = (
        "You are a helpful assistant. You will be given context and based on that you have to answer questions in "
        "every prompt. Don't provide an answer if the information is not in the context."
    )
    SUFFIX = (
        "Develop a Retrieval-Augmented Generation (RAG) system that uses a structured question-answer database as its context. "
        "The system should:\n\n"
        "Input Processing: Accept a user question and preprocess it to correct any spelling errors and clarify ambiguous terms.\n"
        "Contextual Retrieval: Search the question-answer database to find question-answer pairs that are most relevant to the "
        "processed user question. Utilize natural language processing techniques to match the semantics of the question rather than "
        "relying solely on keyword matching.\n"
        "Answer Generation:\n"
        "    If relevant information is available: Use the retrieved question-answer pairs to generate a comprehensive and detailed response. "
        "The answer should integrate all relevant information from the context, ensuring that it addresses all aspects of the user's question. "
        "The system should synthesize the information in a coherent and informative manner.\n"
        "    If no relevant information is available: The system should return 'Answer not available in the context' to indicate that it "
        "cannot provide an accurate answer based on the existing database.\n"
        "Output: Present the answer to the user in a clear and concise format. If multiple question-answer pairs are relevant, synthesize the "
        "information into a single unified response to avoid redundancy and ensure clarity."
    )

    def retrieval_tool(query):
        return "\n".join([f"Question: {doc['content']}\nAnswer: {doc['content']}" for doc in OpenAIops.retriever.retrieve(query)])

    tools = [
        Tool("retrieve", retrieval_tool, description="Retrieves documents based on query embeddings and returns them formatted as question-answer pairs.")
    ]

    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # or any other AgentType as required
        tools=tools,  # Ensure your tools are defined
        llm=llm,  # Your language model
        agent_kwargs={
            'prefix': PREFIX,
            'format_instructions': FORMAT_INSTRUCTIONS,
            'suffix': SUFFIX
        }
    )

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
