#!/usr/bin/env python
# coding: utf-8



# In[2]:


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
import os


# In[3]:


os.environ["GOOGLE_API_KEY"]="AIzaSyDcdj_rMyAJPasM9EX8qPWA7cbnOFro4eM"


# In[6]:


prompt = PromptTemplate.from_template("Answer the following query: {query}")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)
query = "What is the Capital of France?"
response = chain.run(query=query)
print("Chatbot Response:", response)



# In[10]:


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import os


# In[12]:


pdf_path=r"C:\Users\SS458131\Downloads\ABCL010425ABCL-007 Mandatory Leave Guidelines_Final.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()


# In[14]:


#Split Text into Chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks= text_splitter.split_documents(documents)

#Initialize Google AI embeddings for vector store
embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

# Create in-memory vector store (FAISS)
vector_store = FAISS.from_documents(chunks, embedding_model)

#Function to answer multiple questions using RAG

def ask_rag_batch(queries):
    results = {}
    for query in queries:
        # Retrieve relevant documents
        docs = vector_store.similarity_search(query, k=2)
        context ="\n\n".join([doc.page_content for doc in docs])
        
        # Construct the final prompt 
        prompt =f"Use the following document excerpts to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate answer 
        response = llm.invoke(prompt)
        results[query] = response.content.strip()
        
    return results


# In[21]:


# Define batch of questions 
batch_queries = [
    "What is the document about?",
    "What are the key objective of Mandatory Leave guidelines?",
    "Which employees comes under the applicability of Mandatory Leave guidelines?",
    "Summarize this document for m in 3 lines",
    "what is the issue date of this document",
    "WHat are the guidelines?"
     ]

# Run batch RAG query
answers = ask_rag_batch(batch_queries)

#print output
print("\nBatch RAG Answers:")
for query, answer in answers.items():
    print(f"\nQuestion: {query}\nAnswer: {answer}\n")





