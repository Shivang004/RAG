import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq.chat_models import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import dotenv
import os

# Load environment variables
dotenv.load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Kanpur RAG Restaurant Assistant", page_icon="🍽️")
st.title("🍽️ Restaurant RAG Assistant (Kanpur)")
# ---------------------------
# Load and prepare documents
# ---------------------------
@st.cache_resource
def load_vector_store():
    loader = CSVLoader(file_path="restaurants.csv")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

vector_store = load_vector_store()

# ---------------------------
# Initialize LLM and Chains
# ---------------------------
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.7,
    max_tokens=1024
)

basic_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=basic_retriever,
    base_compressor=compressor
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

restaurant_qa_template = """
You are a knowledgeable restaurant recommendation assistant for Kanpur, India.
Answer the question briefly and helpfully based on the information below. Avoid repeating irrelevant context or chat history.

For restaurant recommendations, include if available:
- Name and location
- Price range
- Ratings
- Key features (e.g. outdoor seating, live music, romantic)
- Opening hours
- Contact info

Keep responses concise and directly relevant to the question. Do not mention the source or say things like "based on the data."

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:
"""

QA_PROMPT = PromptTemplate(
    template=restaurant_qa_template, 
    input_variables=["context", "question", "chat_history"]
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compression_retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

query_transform_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Given the following user question about restaurants, reformulate it to be 
    more effective for retrieving relevant information from a database of restaurant details, 
    menus, and features. Just provide the reformulated query without any additional text.

    Original question: {question}
    Improved search query:"""
)

query_transform_chain = LLMChain(
    llm=llm,
    prompt=query_transform_prompt
)

def transform_query(question):
    return query_transform_chain.run(question)



if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a question about restaurants in Kanpur...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        transformed_question = transform_query(user_input)
        result = qa_chain({"question": transformed_question})
        answer = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
