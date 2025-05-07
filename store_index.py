from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

loader=PyPDFLoader("document/Human-behaviour.pdf")
documents=loader.load()

# print("Here is the document : ", documents)

text_chunks=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
# for i in text_chunks:
#     print(i.page_content)
#     print("Page number : ", i.metadata['page'])
#     print("--------------------------------------------------")


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(text_chunks, embedding_model, persist_directory="chroma_db")


db.persist()
print("Database created and persisted successfully.")