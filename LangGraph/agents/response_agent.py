from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import MessageGraph, END
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from .utils_agent import UtilsAgent

class ResponseAgent:
    def __init__(self, db_path):
        openai_api_key = UtilsAgent.get_env_variable("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
        self.memory = SqliteSaver.from_conn_string(db_path)
        self.graph = MessageGraph()
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("generate_response", ToolNode([self.generate_response]))
        self.graph.add_edge("generate_response", END)
        self.graph.set_entry_point("generate_response")

    def generate_response(self, state):
        """Method for generating a response."""

        context = state["context_combination"]

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Develop a Retrieval-Augmented Generation (RAG) system that uses both a structured question-answer database and a Neo4j knowledge graph as its context. The system should:
            Input Processing: Accept a user question and preprocess it to correct any spelling errors and clarify ambiguous terms.
            Contextual Retrieval: Search both the question-answer database and the Neo4j knowledge graph to find relevant information for the processed user question. Utilize natural language processing techniques to match the semantics of the question rather than relying solely on keyword matching.
            Answer Generation: If relevant information is available: Use the retrieved information from both sources to generate a comprehensive and detailed response. The answer should integrate all relevant information from the context, ensuring that it addresses all aspects of the user's question. The system should synthesize the information in a coherent and informative manner.
            If no relevant information is available: The system should return "Answer not available in the context" to indicate that it cannot provide an accurate answer based on the existing sources.
            Output: Output should be presented here. Present the answer to the user in a clear and concise format. If multiple pieces of relevant information are available, synthesize them into a single unified response to avoid redundancy and ensure clarity.
            """),
            ("human", "{context}")
        ])

        prompt = prompt_template.format_prompt(context=context)
        messages = prompt.to_messages()
        response = self.llm(messages)
        output = response.content
        return output
