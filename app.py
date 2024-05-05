# Import necessary libraries for the application
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.chat_models import ChatOpenAI
from flask import Flask, render_template, jsonify, request
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.helper import *
import os
from langchain.schema import Document, BaseRetriever
from typing import List, Any
from pydantic import BaseModel, Field

# Define a custom retriever for document retrieval
class SimpleRetriever(BaseRetriever):
    vector_store: Any  # Store the type as Any or specify if known
    k: int = Field(default=5, description="The number of documents to retrieve")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents based on embeddings similarity, limited to the top k results."""
        return self.vector_store.similarity_search(query=query, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Asynchronously retrieve documents using cosine similarity."""
        return await self.vector_store.similarity_search(query=query, k=self.k)
    

# Initialize the Flask app
app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()

# Retrieve API keys and index name from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Load embeddings for the Pinecone index
embeddings = download__embeddings()

# Initialize Pinecone vector store with an existing index
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
simple_retriever = SimpleRetriever(vector_store=docsearch, k=10)

# Setup prompt template for querying
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the chat model using OpenAI's GPT-4
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.8
)

# Create a RetrievalQA object with specified chain type and retriever
qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  # Ensure this is a valid chain type or replace as necessary
    retriever=simple_retriever,
    chain_type_kwargs=chain_type_kwargs 
)  

# Define the home page route
@app.route("/")
def index():
    # Serve the main HTML page
    return render_template('brazil.html')

# Define a function to process input from the web interface
def process(input:str):
    # Run the query against the QA system
    return qa({"query": input})

# Define the chat endpoint to handle messages
@app.route("/get", methods=["GET", "POST"])
def chat():
    # Retrieve message from form data
    msg = request.form["msg"]
    input = msg
    print(input)
    # Process the input message
    result = process(input)
    print("Response : ", result["result"])
    # Return the result as a string
    return str(result["result"])

# Run the Flask application on the specified host and port
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
