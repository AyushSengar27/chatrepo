from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone 
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.chat_models import ChatOpenAI
from flask import Flask, render_template, jsonify, request
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.document_loaders import CSVLoader
from dotenv import load_dotenv
from src.prompt import *
from src.helper import *
import os
from langchain.schema import Document, BaseRetriever
from typing import List, Any
from pydantic import BaseModel, Field

class SimpleRetriever(BaseRetriever):
    vector_store: Any  # Specify the correct type if known, else use `Any`
    k: int = Field(default=5, description="The number of documents to retrieve")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents based on embeddings similarity, limited to the top k results using cosine similarity."""
        return self.vector_store.similarity_search(query=query, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Asynchronously retrieve documents using cosine similarity."""
        return await self.vector_store.similarity_search(query=query, k=self.k)
    


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


embeddings = download__embeddings()


index_name=index_name

#Loading the index
docsearch=PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings)
simple_retriever = SimpleRetriever(vector_store=docsearch, k=10)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.8
)


qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=simple_retriever,
    chain_type_kwargs=chain_type_kwargs 
)  



@app.route("/")
def index():
    return render_template('brazil.html')


def process(input:str):
    return qa({"query": input})



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=process(input)
    print("Response : ", result["result"])
    return str(result["result"])






if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)