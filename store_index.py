from src.helper import load_csv, text_split, download__embeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone 
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Pinecone
import pinecone
from pathlib import Path
from langchain.document_loaders import CSVLoader
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX')


# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_csv('data\data.csv')
text_chunks = text_split(extracted_data)
embeddings = download__embeddings()


index_name=index_name

docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], index_name=index_name,embedding=embeddings)