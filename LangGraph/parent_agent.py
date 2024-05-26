from langgraph.graph import MessageGraph, END
from langchain.schema import HumanMessage
from agents.database_agent import DatabaseAgent
from agents.query_generation_agent import QueryGenerationAgent
from agents.response_agent import ResponseAgent
from agents.embedding_agent import EmbeddingAgent
from agents.graph_agent import GraphAgent
from tools.embedding_retriever import EmbeddingRetriever
from agents.utils_agent import UtilsAgent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


def ParentAgent(db_path, state):

    utils_agent = UtilsAgent(db_path=db_path)
    embedding_agent = EmbeddingAgent(db_path=db_path)
    database_agent = DatabaseAgent(db_path=db_path)
    graph_agent = GraphAgent(db_path=db_path)
    query_generation_agent = QueryGenerationAgent(db_path=db_path)
    response_agent = ResponseAgent(db_path=db_path)

    memory = SqliteSaver.from_conn_string(db_path)
    graph = MessageGraph()

    # Step 1: Receive User Query
    user_query_node = state["user_query"]
    graph.add_node("user_query", user_query_node)

    # Step 2: Setup Utility Handling
    graph = utils_agent.graph
    graph.add_node("getting_environment_variables", utils_agent.get_env_variable)
    graph.add_node("loading_db_credentials", utils_agent.load_db_credentials)

    # Step 3: Setup Embedding Generation
    graph = embedding_agent.graph
    graph.add_node("embedding_generation_for_docs", embedding_agent.embed_documents)
    graph.add_node("embedding_generation_for_queries", embedding_agent.embed_query)

    # Step 4: Database Operations
    graph = database_agent.graph
    graph.add_node("execute_database_ops", database_agent.execute(state["data_csv"]))

    # Step 5: Knowledge Graph Operations
    graph = graph_agent.graph
    graph.add_node("create_knowledge_graph", graph_agent.create_knowledge_graph(state["data_csv"]))

    # Step 6: Database Retrieval
    def database_retrieval_function(state):
        """Method for database retrieval."""
        with database_agent.db_ops.get_database_connection() as db_connection:
            embedding_retriever = EmbeddingRetriever(db_connection=db_connection, db_path=db_path, embeddings=embedding_agent.embeddings)
            return embedding_retriever.get_relevant_documents(state["user_query"])

    database_retrieval_node = ToolNode([database_retrieval_function])
    graph.add_node("database_retrieval", database_retrieval_node)

    # Step 7: Query Generation
    def query_generation_function(state):
        """Method for query generation."""
        return query_generation_agent.generate_cypher_query(state["user_query"])

    query_generation_node = ToolNode([query_generation_function])
    graph.add_node("query_generation", query_generation_node)

    # Step 8: Graph Retrieval
    graph = graph_agent.graph
    graph.add_node("graph_retrieval", graph_agent.query_knowledge_graph(state["user_query"]))

    # Step 9: Context Combination
    def context_combination_function(state):
        """Method for context combination."""
        return {
            "user_query": state["user_query"],
            "database_context": state["database_retrieval"],
            "graph_context": state["graph_retrieval"]
        }

    context_combination_node = ToolNode([context_combination_function])
    graph.add_node("context_combination", context_combination_node)

    # Step 10: Response Generation
    graph = response_agent.graph
    graph.add_node("response_generation", response_agent.generate_response(state["context_combination"]))

    # Define edges
    graph.add_edge("user_query", "getting_environment_variables")
    graph.add_edge("getting_environment_variables", "loading_db_credentials")
    graph.add_edge("loading_db_credentials", "embedding_generation_for_docs")
    graph.add_edge("embedding_generation_for_docs", "embedding_generation_for_queries")
    graph.add_edge("embedding_generation_for_queries", "execute_database_ops")
    graph.add_edge("execute_database_ops", "create_knowledge_graph")
    graph.add_edge("create_knowledge_graph", "database_retrieval")
    graph.add_edge("database_retrieval", "query_generation")
    graph.add_edge("query_generation", "graph_retrieval")
    graph.add_edge("graph_retrieval", "context_combination")
    graph.add_edge("context_combination", "response_generation")
    graph.add_edge("response_generation", END)

    start = graph.set_entry_point("user_query")
    runnable = graph.compile(checkpointer=memory)
    runnable.invoke(HumanMessage(start))

    return runnable
