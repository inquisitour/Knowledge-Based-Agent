import os
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from data_processing import get_database_connection
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Securely fetch the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

class EmbeddingRetriever:
    def __init__(self, db_connection):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
        self.db_connection = db_connection
        print("Embedding retriever initialized")

    def retrieve_similar_questions(self, query, k=5):
        query_vec = self.embeddings.embed_documents(query)[0]
        query_vec = np.array(query_vec) if isinstance(query_vec, list) else query_vec
        similar_questions = []
        with self.db_connection.cursor() as cursor:
            cursor.execute("SELECT question, answer, embedding FROM faq_embeddings")
            results = cursor.fetchall()
            for result in results:
                question, answer, embedding = result
                embedding = np.frombuffer(embedding, dtype=np.float32)
                similarity = np.dot(embedding, query_vec) / (np.linalg.norm(embedding) * np.linalg.norm(query_vec))
                similar_questions.append({'question': question, 'answer': answer, 'similarity': similarity})
            similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_questions[:k]

class OpenAIops:
    def __init__(self):
        self.chat_model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')
        with get_database_connection() as conn:
            self.retriever = EmbeddingRetriever(conn)
        print("OpenAI operations initialized")

    def answer_question(self, user_question):
        context = self.retriever.retrieve_similar_questions(user_question)
        context = "\n\n".join([f"Q: {q['question']}, A: {q['answer']}" for q in context])
        prompt_template = "Context:\n"+context+"\n\n\nQuestion: \n"+user_question+"\n Answer:"
        sysmsg = """Develop a Retrieval-Augmented Generation (RAG) system that uses a structured question-answer database as its context. The system should:
                    Input Processing: Accept a user question and preprocess it to correct any spelling errors and clarify ambiguous terms.
                    Contextual Retrieval: Search the question-answer database to find question-answer pairs that are most relevant to the processed user question. Utilize natural language processing techniques to match the semantics of the question rather than relying solely on keyword matching.
                    Answer Generation:
                    If relevant information is available: Use the retrieved question-answer pairs to generate a comprehensive and detailed response. The answer should integrate all relevant information from the context, ensuring that it addresses all aspects of the user's question. The system should synthesize the information in a coherent and informative manner.
                    If no relevant information is available: The system should return "Answer not available in the context" to indicate that it cannot provide an accurate answer based on the existing database.
                    Output: Output should be preseted here. Present the answer to the user in a clear and concise format. If multiple question-answer pairs are relevant, synthesize the information into a single unified response to avoid redundancy and ensure clarity. """

        messages = [
            SystemMessage(content=sysmsg),
            HumanMessage(content=prompt_template)
        ]

        context = self.chat_model(messages=messages) 
        print("Finishing up..!")
        output_section = context.content
        
        # Search for both 'Answer:' and 'Output:'
        answer_index = output_section.find('Answer:')
        output_index = output_section.find('Output:')

        # Determine which index to use (use the first valid index found)
        if answer_index != -1 and (output_index == -1 or answer_index < output_index):
            output_section = output_section[answer_index + len('Answer:'):].strip()
        elif output_index != -1:
            output_section = output_section[output_index + len('Output:'):].strip()
            
        return output_section

class ResponseAgent:
    def __init__(self):
        self.openaiops = OpenAIops()
        print("Response agent initialized")

    def answer_question(self, user_question):
        return self.openaiops.answer_question(user_question)
