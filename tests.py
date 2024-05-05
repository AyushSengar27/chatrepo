import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.helper import *
from langchain.schema import Document, BaseRetriever
from typing import List, Any
from pydantic import BaseModel, Field
from openai import OpenAI

# Define a simple document retriever class to fetch relevant documents based on a query
class SimpleRetriever(BaseRetriever):
    vector_store: Any  # Specify the correct type if known, else use `Any`
    k: int = Field(default=5, description="The number of documents to retrieve")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents based on embeddings similarity, limited to the top k results using cosine similarity."""
        return self.vector_store.similarity_search(query=query, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Asynchronously retrieve documents using cosine similarity."""
        return await self.vector_store.similarity_search(query=query, k=self.k)
    



load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = download__embeddings()


index_name=index_name

#Loading the index
docsearch=PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings)
# Initialize the retriever with the Pinecone vector store and a parameter k
simple_retriever = SimpleRetriever(vector_store=docsearch, k=10)

# Define the prompt template and chain type arguments for the retrieval QA model
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

# Initialize the Chat model using OpenAI with specified model parameters
llm=ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.8
)

# Create a RetrievalQA object which uses the LangChain library for a question-answering setup
qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=simple_retriever,
    chain_type_kwargs=chain_type_kwargs 
)  

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response convey similar message as expected response? 
"""
# Define a function to evaluate the response against an expected answer
def query_and_validate(question: str, expected_response: str):
    result= qa({"query": question})
    response_text = str(result["result"])
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    try:
        # Call OpenAI's GPT-4 model
        response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=64,
        top_p=1
    )
        evaluation_results_str = response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(response)
        print(f"An error occurred: {e}")
        return False

    

    if "true" in evaluation_results_str:
        print(f"Response: {evaluation_results_str}")
        return True
    elif "false" in evaluation_results_str:
        print(f"Response: {evaluation_results_str}")
        print(response_text)
        return False
    else:
        raise ValueError("Invalid evaluation result. Cannot determine if 'true' or 'false'.")


def test_internet_users():
    assert query_and_validate(
        question="internet users in 2020",
        expected_response="81.34% population in Brazil were using the Internet.",
    )


def test_poverty_data_source():
    assert query_and_validate(
        question="which organization collects data for poverty",
        expected_response="The data for poverty is collected by the World Bank, specifically their Poverty and Inequality Platform.",
    )


