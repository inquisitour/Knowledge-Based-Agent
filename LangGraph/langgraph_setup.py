from langgraph.graph import MessageGraph, END
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from agents.database_agent import DatabaseAgent
from agents.query_generation_agent import QueryGenerationAgent
from agents.response_agent import ResponseAgent
from agents.embedding_agent import EmbeddingAgent
from tools.graph_embedding_retriever import GraphEmbeddingRetriever
from tools.embedding_retriever import EmbeddingRetriever
from agents.utils_agent import load_db_credentials, UtilsAgent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

def setup_langgraph(openai_api_key, db_path):
    utils_agent = UtilsAgent(db_path=db_path)
    db_config = utils_agent.process_load_db_credentials('postgres')
    neo4j_credentials = utils_agent.process_load_db_credentials('neo4j')

    database_agent = DatabaseAgent(db_path=db_path)
    query_generation_agent = QueryGenerationAgent(db_path=db_path)
    response_agent = ResponseAgent(db_path=db_path)
    embedding_agent = EmbeddingAgent(db_path=db_path)
    
    graph_embedding_retriever = GraphEmbeddingRetriever(
        neo4j_credentials['uri'],
        neo4j_credentials['username'],
        neo4j_credentials['password'],
        openai_api_key,
        db_path
    )
    embedding_retriever = EmbeddingRetriever(db_connection=database_agent.db_connection, db_path=db_path)

    memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    graph = MessageGraph(memory=memory)

    # Step 1: Receive User Query
    def user_query_function(query):
        return {"user_query": query}

    user_query_node = ToolNode(user_query_function)
    graph.add_node("user_query", user_query_node)

    # Step 2: Database Retrieval
    def database_retrieval_function(state):
        return database_agent.get_relevant_documents(state["user_query"])

    database_retrieval_node = ToolNode(database_retrieval_function)
    graph.add_node("database_retrieval", database_retrieval_node)

    # Step 3: Graph Retrieval
    def graph_retrieval_function(state):
        return graph_embedding_retriever.query_knowledge_graph(state["user_query"])

    graph_retrieval_node = ToolNode(graph_retrieval_function)
    graph.add_node("graph_retrieval", graph_retrieval_node)

    # Step 4: Query Generation
    def query_generation_function(state):
        return query_generation_agent.generate_cypher_query(state["user_query"])

    query_generation_node = ToolNode(query_generation_function)
    graph.add_node("query_generation", query_generation_node)

    # Step 5: Context Combination and Response Generation
    def context_combination_function(state):
        return {
            "database_context": state["database_retrieval"],
            "graph_context": state["graph_retrieval"],
            "generated_query": state["query_generation"]
        }

    context_combination_node = ToolNode(context_combination_function)
    graph.add_node("context_combination", context_combination_node)

    def response_generation_function(state):
        return response_agent.generate_response(state["context_combination"])

    response_generation_node = ToolNode(response_generation_function)
    graph.add_node("response_generation", response_generation_node)

    # Define edges
    graph.add_edge("user_query", "database_retrieval")
    graph.add_edge("user_query", "graph_retrieval")
    graph.add_edge("user_query", "query_generation")
    graph.add_edge("database_retrieval", "context_combination")
    graph.add_edge("graph_retrieval", "context_combination")
    graph.add_edge("query_generation", "context_combination")
    graph.add_edge("context_combination", "response_generation", END)

    return graph

# Example usage:
if __name__ == "__main__":
    openai_api_key = UtilsAgent.process_get_env_variable("OPENAI_API_KEY")
    db_path = "langgraph_memory.db"
    langgraph = setup_langgraph(openai_api_key, db_path)

    user_query = "Tell me about AI."
    result = langgraph.run("user_query", query=user_query)
    print(result)
