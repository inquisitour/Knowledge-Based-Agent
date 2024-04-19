import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SimpleRetriever
from langchain.agents import Agent, Tool, ToolKit
from data_preprocessing import DBops

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

class ResponseAgent(Agent):
    def __init__(self, chat_model, retriever):
        self.chat_model = chat_model
        self.retriever = retriever
        tools = [
            Tool("retrieve", self.get_context),
            Tool("generate_response", self.generate_response)
        ]
        self.toolkit = ToolKit(tools)

    def get_context(self, questions):
        documents = self.retriever.retrieve(questions)
        return "\n".join([f"Question: {doc['content'][0]}\nAnswer: {doc['content'][1]}" for doc in documents])

    def generate_response(self, context, prompt):
        response = self.chat_model.chat(prompt + "\n\n" + context, max_tokens=150)
        return response

    def perform(self, user_question):
        user_embedding = OpenAIops.get_embeddings([user_question])[0]
        similar_questions = DBops.get_similar_questions(user_embedding)
        context = self.get_context(similar_questions)

        sysmsg1 = "Please use the information from the previous questions and answers to help."
        sysmsg2 = "Ensure all relevant details are included and any irrelevant information is excluded."
        prompt_template = (sysmsg1 + "\n" + sysmsg2 +
                           "\nAnswer the question as detailed as possible from the provided context "
                           "which can be info from the questions or the answers. Make sure to provide all the details, "
                           "account for spelling errors assume the closest meaning where unclear. Use concise and clear language.")
        
        return self.generate_response(context, prompt_template)

class OpenAIops:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    chat_model = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-3.5-turbo'
    )
    retriever = SimpleRetriever(embeddings)
    agent = ResponseAgent(chat_model, retriever)

    @staticmethod
    def get_embeddings(questions):
        return OpenAIops.embeddings.embed_documents(questions)

    @staticmethod
    def answer_question(user_question):
        return OpenAIops.agent.perform(user_question)
