import os
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from data_processing import get_database_connection
from typing import List, Any
from langchain.schema import Document, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from pydantic import BaseModel, Field
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Securely fetch the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

# Set environment variables
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "gravitas@123"

# Initialize the Neo4j Graph connection
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))

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

                # Reshape the embeddings to match dimensionality
                #if len(embedding) < len(query_vec):
                    #embedding = np.pad(embedding, (0, len(query_vec) - len(embedding)), mode='constant')
                #elif len(embedding) > len(query_vec):
                    #query_vec = np.pad(query_vec, (0, len(embedding) - len(query_vec)), mode='constant')

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
        self.chat_model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')
        with get_database_connection() as conn:
            self.retriever = EmbeddingRetriever(conn)
            
            # Convert the retriever into a LangChain tool
            retriever_tool = Tool(
                name="EmbeddingRetriever",
                func=self.retriever.get_relevant_documents,  
                description="Retrieves similar questions and answers from the database"
            )

            # Define the prompt template
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", """Develop a Retrieval-Augmented Generation (RAG) system that uses a structured question-answer database as its context. The system should:
                Input Processing: Accept a user question and preprocess it to correct any spelling errors and clarify ambiguous terms.
                Contextual Retrieval: Search the question-answer database to find question-answer pairs that are most relevant to the processed user question. Utilize natural language processing techniques to match the semantics of the question rather than relying solely on keyword matching.
                Answer Generation: If relevant information is available: Use the retrieved question-answer pairs to generate a comprehensive and detailed response. The answer should integrate all relevant information from the context, ensuring that it addresses all aspects of the user's question. The system should synthesize the information in a coherent and informative manner.
                If no relevant information is available: The system should return "Answer not available in the context" to indicate that it cannot provide an accurate answer based on the existing database.
                Output: Output should be presented here. Present the answer to the user in a clear and concise format. If multiple question-answer pairs are relevant, synthesize the information into a single unified response to avoid redundancy and ensure clarity. """
                ),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])

            # Initialize the agent with the tool
            tools = [retriever_tool]

            # Initialize the agent with the tool
            self.agent = create_openai_tools_agent(
                llm=self.chat_model,
                tools=tools,
                prompt=self.prompt_template
            )
            self.agent_executor = AgentExecutor.from_agent_and_tools(self.agent, tools)
        print("OpenAI operations with LangChain agent initialized")

    def answer_question(self, user_question):
        '''# Use LangChain to process the user query and generate a Cypher query
        instructions = """Given the query, consider following schema used to create the knowledge graph:
                    MERGE (c:Category {name: $category})
                    MERGE (q:Question {id: $index, text: $question, category: $category})
                    MERGE (a:Answer {id: $index, text: $answer, category: $category})
                    MERGE (q)-[:HAS_ANSWER]->(a)
                    MERGE (c)-[:INCLUDES]->(q)
                    MERGE (c)-[:INCLUDES]->(a)
                    and generate a Cypher query to retrieve the relevant information from the Neo4j knowledge graph. Ensure that the generated query only use and adheres to this schema and retrieves the desired information accurately. No other things must be generated than Cypher query."""
        messages = HumanMessage(content=user_question + instructions)
        query = self.chat_model([messages])
        print(query)
        cypher_query = query.content
        results = graph.query(cypher_query)
        print(results)'''

        context = self.retriever.get_relevant_documents(user_question)
        formatted_context = "\n\n".join([f"Q: {doc.metadata['question']}, A: {doc.page_content}" for doc in context])
        prompt = f"Context:\n{formatted_context}\n\nQuestion: \n{user_question}\nAnswer:"
        #prompt = f"Knowledge Graph:\n{results}\n\nContext:\n{formatted_context}\n\nQuestion: \n{user_question}\nAnswer:"

        # Execute the agent with the dynamically formatted prompt
        response = self.agent_executor({"input": prompt}) 
        output = response["output"]
        print("Finishing up..!")
            
        return output

class ResponseAgent:
    def __init__(self):
        self.openaiops = OpenAIops()
        print("Response agent initialized")

    def answer_question(self, user_question):
        return self.openaiops.answer_question(user_question)