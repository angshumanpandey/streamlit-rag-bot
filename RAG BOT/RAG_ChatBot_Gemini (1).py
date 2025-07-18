import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import os

# Set page config
st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")

# Title
st.title("📄🔍 Gemini RAG ChatBot")

# Get Google API key from user securely
api_key = st.text_input("Enter your Google Generative AI API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file is not None and api_key:

    # Save uploaded file temporarily
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF content
    loader = PyPDFLoader("uploaded_pdf.pdf")
    pages = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    # Embeddings and vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # User query
    query = st.text_input("Ask a question from the PDF")

    if query:
        docs_with_scores = vectorstore.similarity_search(query, k=3)

        # Load Gemini model and QA chain
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        chain = load_qa_chain(llm, chain_type="stuff")

        # Get answer
        response = chain.run(input_documents=docs_with_scores, question=query)

        # Show response
        st.markdown("### 💬 Answer")
        st.write(response)

elif uploaded_file is None and api_key:
    st.warning("Please upload a PDF file to continue.")
elif uploaded_file and not api_key:
    st.warning("Please enter your Google Generative AI API key.")
