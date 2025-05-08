from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
load_dotenv() 

key =os.getenv("GOOGLE_API_KEY")

query=input("Enter your query: ")
print("You entered:", query)


prompt_template = """You are a helpful assistant for answering questions about documents.
Use the following context to answer the question. If you don't know, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])



embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model )



llm=ChatGoogleGenerativeAI(temperature=0.3, model="gemini-2.0-flash", key=key)


qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever() , chain_type="stuff", chain_type_kwargs={"prompt": PROMPT},return_source_documents=True) 

# Ask a question
response = qa_chain.invoke({ "query": query})

print("Answer:", response["result"])
print("Sources:")
for doc in response["source_documents"]:
    print(doc.metadata.get("source", "Unknown"), "-", doc.page_content[:100])
