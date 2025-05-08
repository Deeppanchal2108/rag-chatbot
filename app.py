import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

# Set up Streamlit UI
st.set_page_config(page_title="📄 RAG PDF Chatbot", layout="centered")
st.title("📄 RAG PDF Chatbot")
query = st.text_input("Ask a question about your documents:")

# Define prompt
prompt_template = """You are a helpful assistant for answering questions about documents.
Use the following context to answer the question. If you don't know, say you don't know.

Context:
{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Embeddings and Vector DB
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# LLM
llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-2.0-flash", key=key)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Run on question
if query:
    response = qa_chain.invoke({"query": query})
    st.subheader("Answer:")
    st.write(response["result"])

    st.subheader("Sources:")
    for doc in response["source_documents"]:
        source = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:200]
        st.markdown(f"**{source}** - {preview}...")
