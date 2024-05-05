from src.helper import load_csv, text_split, download__embeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from pathlib import Path
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