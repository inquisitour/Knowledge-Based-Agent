from langgraph.graph import MessageGraph, END
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from agents.database_agent import DatabaseAgent
from agents.query_generation_agent import QueryGenerationAgent
from agents.response_agent import ResponseAgent
from agents.embedding_agent import EmbeddingAgent
from tools.graph_embedding_retriever import GraphEmbeddingRetriever
from tools.embedding_retriever import EmbeddingRetriever
from agents.utils_agent import UtilsAgent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

model = ChatOpenAI(temperature=0)

def AgentState(openai_api_key, db_path):
    utils_agent = UtilsAgent(db_path=db_path)
    database_agent = DatabaseAgent(db_path=db_path)
    query_generation_agent = QueryGenerationAgent(db_path=db_path)
    response_agent = ResponseAgent(db_path=db_path)
    embedding_agent = EmbeddingAgent(db_path=db_path)

    memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    graph = MessageGraph(memory=memory)

    # Step 1: Receive User Query
    def user_query_function(query):
        return {"user_query": query}

    user_query_node = ToolNode(user_query_function)
    graph.add_node("user_query", user_query_node)

    # Step 2: Utility Handling
    utils_graph = utils_agent.get_graph()
    graph.add_node("utility_handling", utils_graph)

    # Step 3: Embedding Generation
    embedding_graph = embedding_agent.get_graph()
    graph.add_node("embedding_generation", embedding_graph)

    # Step 4: Database Operations
    db_ops_graph = database_agent.get_graph(embedding_agent.embeddings)
    graph.add_node("database_ops", db_ops_graph)

    # Step 5: Create Knowledge Graph
    def create_knowledge_graph_function(state):
        neo4j_credentials = state["neo4j_credentials"]
        graph_embedding_retriever = GraphEmbeddingRetriever(
            neo4j_credentials['uri'],
            neo4j_credentials['username'],
            neo4j_credentials['password'],
            openai_api_key,
            db_path,
            embeddings=embedding_agent.embeddings
        )
        data_csv = state["data_csv"]  # Assuming the CSV data is available in the state
        graph_embedding_retriever.process_create_knowledge_graph(data_csv)

    create_knowledge_graph_node = ToolNode(create_knowledge_graph_function)
    graph.add_node("create_knowledge_graph", create_knowledge_graph_node)

    # Step 6: Database Retrieval
    def database_retrieval_function(state):
        with database_agent.db_ops.get_database_connection() as db_connection:
            embedding_retriever = EmbeddingRetriever(db_connection=db_connection, db_path=db_path, embeddings=embedding_agent.embeddings)
            return embedding_retriever.get_relevant_documents(state["user_query"])

    database_retrieval_node = ToolNode(database_retrieval_function)
    graph.add_node("database_retrieval", database_retrieval_node)

    # Step 7: Query Generation
    def query_generation_function(state):
        return query_generation_agent.generate_cypher_query(state["user_query"])

    query_generation_node = ToolNode(query_generation_function)
    graph.add_node("query_generation", query_generation_node)

    # Step 8: Graph Retrieval
    def graph_retrieval_function(state):
        neo4j_credentials = state["neo4j_credentials"]
        graph_embedding_retriever = GraphEmbeddingRetriever(
            neo4j_credentials['uri'],
            neo4j_credentials['username'],
            neo4j_credentials['password'],
            openai_api_key,
            db_path,
            embeddings=embedding_agent.embeddings
        )
        return graph_embedding_retriever.process_query_knowledge_graph(state["user_query"], state["generated_query"])

    graph_retrieval_node = ToolNode(graph_retrieval_function)
    graph.add_node("graph_retrieval", graph_retrieval_node)

    # Step 9: Context Combination
    def context_combination_function(state):
        return {
            "database_context": state["database_retrieval"],
            "graph_context": state["graph_retrieval"],
            "generated_query": state["query_generation"]
        }

    context_combination_node = ToolNode(context_combination_function)
    graph.add_node("context_combination", context_combination_node)

    # Step 10: Response Generation
    response_generation_graph = response_agent.get_graph()
    graph.add_node("response_generation", response_generation_graph)

    # Define edges
    graph.add_edge("user_query", "utility_handling")
    graph.add_edge("utility_handling", "embedding_generation")
    graph.add_edge("embedding_generation", "database_ops")
    graph.add_edge("database_ops", "create_knowledge_graph")
    graph.add_edge("create_knowledge_graph", "database_retrieval")
    graph.add_edge("database_retrieval", "query_generation")
    graph.add_edge("query_generation", "graph_retrieval")
    graph.add_edge("graph_retrieval", "context_combination")
    graph.add_edge("database_retrieval", "context_combination")
    graph.add_edge("context_combination", "response_generation")
    graph.add_edge("response_generation", END)

    start = graph.set_entry_point("user_query")
    runnable = graph.compile()
    runnable.invoke(HumanMessage(start))

    return runnable

# Example usage:
if __name__ == "__main__":
    openai_api_key = UtilsAgent.process_get_env_variable("OPENAI_API_KEY")
    db_path = "langgraph_memory.db"
    langgraph = AgentState(openai_api_key, db_path)

    user_query = "Tell me about AI."
    result = langgraph.run("user_query", query=user_query)
    print(result)