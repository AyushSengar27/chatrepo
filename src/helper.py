from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader

def load_csv(file_path):
    # Use 'utf-8' encoding when reading the CSV file
    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    documents = loader.load()
    return documents




#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download__embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings