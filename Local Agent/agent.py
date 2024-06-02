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
from neo4j_LangChain_Test import GraphEmbeddingRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Securely fetch the API key
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
        return documents

class OpenAIops:
    def __init__(self):
        self.chat_model = ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-3.5-turbo')
        print("before database con")
        with get_database_connection() as conn:
            print("database con established")
            self.retriever = EmbeddingRetriever(conn)
            
            # Convert the retriever into a LangChain tool
            retriever_tool = Tool(
                name="EmbeddingRetriever",
                func=self.retriever.get_relevant_documents,  
                description="Retrieves similar questions and answers from the database"
            )

        self.graph_retriever = GraphEmbeddingRetriever(
            # neo4j_uri="bolt://localhost:7687",
            # neo4j_username="neo4j",
            # neo4j_password="gravitas@123",
            # openai_api_key=OPENAI_API_KEY
        )

        self.graph_retriever.load_knowledge_graph()
        
        print("load knnowledge graph done")

        # Convert the GraphEmbeddingRetriever into a LangChain tool
        graph_retriever_tool = Tool(
            name="GraphEmbeddingRetriever",
            func=self.graph_retriever.query_knowledge_graph,
            description="Retrieves relevant information from the Neo4j knowledge graph [the input must be in english if in hinglish convert it to english first]"
        )

        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Develop a Retrieval-Augmented Generation (RAG) system that uses both a structured question-answer database and a Neo4j knowledge graph as its context. The system should:
            Input Processing: Accept a user question and preprocess it to correct any spelling errors and clarify ambiguous terms.
            Contextual Retrieval: Search both the question-answer database and the Neo4j knowledge graph to find relevant information for the processed user question. Utilize natural language processing techniques to match the semantics of the question rather than relying solely on keyword matching.
            Answer Generation: If relevant information is available: Use the retrieved information from both sources to generate a comprehensive and detailed response. The answer should integrate all relevant information from the context, ensuring that it addresses all aspects of the user's question. The system should synthesize the information in a coherent and informative manner.
            If no relevant information is available: The system should return "Answer not available in the context" to indicate that it cannot provide an accurate answer based on the existing sources.
            Output: Output should be presented here. Present the answer to the user in a clear and concise format. If multiple pieces of relevant information are available, synthesize them into a single unified response to avoid redundancy and ensure clarity. """
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Tools for the agent
        tools = [retriever_tool, graph_retriever_tool]

        # Initialize the agent with the tools
        self.agent = create_openai_tools_agent(
            llm=self.chat_model,
            tools=tools,
            prompt=self.prompt_template
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(self.agent, tools)
        print("OpenAI operations with LangChain agent initialized")

    def answer_question(self, user_question):
        print("in answer    question")
        # context = self.retriever.get_relevant_documents(user_question)
        # print(context)
        # -----------------
        # graph_context = self.graph_retriever.query_knowledge_graph(user_question)
        # formatted_context = ""
        # # formatted_context = "\n\n".join([f"Q: {doc.metadata['question']}, A: {doc.page_content}" for doc in context])
        # formatted_graph_context = "\n\n".join([f"Reference Question : {result['question']} (Correct Answer: {result['answer']})" for result in graph_context])
        # prompt = f"Context:\n{formatted_context}\n\nKnowledge Graph Context:\n{formatted_graph_context}\n\nQuestion: \n{user_question}\nAnswer:"
        # print("Prompt:", prompt)
        # ---------------------
        # Execute the agent with the dynamically formatted prompt
        response = self.agent_executor({"input": user_question}) 
        output = response["output"]
        print("Finishing up..!")
            
        return output

class ResponseAgent:
    def __init__(self):
        self.openaiops = OpenAIops()
        print("Response agent initialized")

    def answer_question(self, user_question):
        return self.openaiops.answer_question(user_question)