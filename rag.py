import streamlit as st
import ast
from langchain.schema import Document
import pandas as pd
import os
import dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq.chat_models import ChatGroq
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage

# Page configuration
st.set_page_config(
    page_title="Kanpur Restaurant Finder",
    page_icon="🍽️",
    layout="wide"
)

# Header section
st.title("🍽️ Kanpur Restaurant Recommendation Assistant")
st.markdown("""
This app helps you find the perfect restaurant in Kanpur based on your preferences.
Ask about cuisines, menu items, prices, locations, or compare restaurants!
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for data loading and model setup
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False

# Function to load environment variables
@st.cache_resource
def load_env_vars():
    dotenv.load_dotenv()
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    return True

# Function to load data and setup models
@st.cache_resource
def load_data_and_models():
    with st.spinner("Loading restaurant data and setting up models..."):
        # ------------------ Step A: CSV Load ------------------
        try:
            csv_path = "merged_data.csv"
            df = pd.read_csv(csv_path)
            
            # ------------------ Step B: Parse & Clean Menu JSON Strings ------------------
            def parse_menu(menu_str):
                try:
                    return ast.literal_eval(menu_str) if pd.notnull(menu_str) else []
                except Exception as e:
                    print(f"Error parsing menu: {menu_str} \n{e}")
                    return []
            
            df['Parsed_Menu'] = df['Menu'].apply(parse_menu)
            
            # ------------------ Step C: Rich Text Construction with Menu Items ------------------
            def build_rich_text(row):
                name = row.get('Restaurant Name', 'Unknown')
                location = row.get('Location', '')
                contact = row.get('Contact', '')
                cuisine = row.get('Cuisine', '')
                price = row.get('Price', '')
                ratings = row.get('Ratings(Dining, Delivery)', '')
                hours = row.get('Operating Hours', '')
                info = row.get('More Info', '')
            
                # Format ratings
                rating_dining, rating_delivery = "N/A", "N/A"
                try:
                    ratings_list = ast.literal_eval(ratings)
                    if isinstance(ratings_list, list) and len(ratings_list) == 2:
                        rating_dining, rating_delivery = ratings_list
                except:
                    pass
            
                menu_text = ". ".join([f"{item['item']} for {item['price']}" for item in row['Parsed_Menu']])
            
                full_text = (
                    f"{name} is located at {location}. "
                    f"Cuisine: {cuisine}. Price range: {price}. "
                    f"Ratings - Dining: {rating_dining}, Delivery: {rating_delivery}. "
                    f"Contact: {contact}. Operating Hours: {hours}. "
                    f"Additional Info: {info}. "
                    f"Menu includes: {menu_text}."
                )
                return full_text
            
            df['rich_text'] = df.apply(build_rich_text, axis=1)
            
            # ------------------ Step D: LangChain Document Creation ------------------
            documents = [
                Document(
                    page_content=row['rich_text'],
                    metadata={
                        "restaurant_name": row['Restaurant Name'],
                        "location": row['Location']
                    }
                )
                for _, row in df.iterrows()
            ]
            
            # Initialize embeddings - using a strong open-source model
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Create vector store
            vector_store = FAISS.from_documents(documents, embeddings)
            
            # Initialize the Groq LLM
            llm = ChatGroq(
                model_name="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024
            )
            
            # Basic retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Initialize memory
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
            
            # Update your chain with this prompt
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": QA_PROMPT}
            )
            
            def get_recent_chat_history(n=3):
                # Get all messages
                history = memory.chat_memory.messages
                # Only keep the last n pairs (each turn is user + assistant = 2 messages)
                last_n = history[-(n*2):] if len(history) >= n*2 else history
                return "\n".join([f"{m.type.capitalize()}: {m.content}" for m in last_n if isinstance(m, BaseMessage)])
            
            # --------------------------------------
            # Intent Classifier
            # --------------------------------------
            intent_prompt = PromptTemplate.from_template("""
            You are an intent classifier for a restaurant chatbot. Given a user query, classify it into one of the following intents:
            - "comparison": Only if the user wants to compare two or more restaurants
            - "diet": if the user is asking about dietary needs, pricing, veg/non-veg, or food restrictions
            - "qa": if the user is asking about restaurant features, menus, ratings, or contact info
            - "fallback": if the question is unrelated or unclear
            
            Respond with only one word: comparison, diet, qa, or fallback. Before classifying as fallback, check the chat history for context.    
                                                         
            Chat History:
            {chat_history}
            
            User Query: "{question}"
            Intent:
            """)
            
            intent_chain = LLMChain(llm=llm, prompt=intent_prompt)
            
            def classify_intent(question: str) -> str:
                chat_his = get_recent_chat_history(3)
                intent = intent_chain.run({"question": question, "chat_history": chat_his}).strip().lower()
                valid_intents = {"comparison", "diet", "qa", "fallback"}
                return intent if intent in valid_intents else "fallback"
            
            # --------------------------------------
            # Comparison Tool (Structured Prompt)
            # --------------------------------------
            compare_prompt = PromptTemplate.from_template("""
            You are a restaurant analyst comparing menus and features. Answer based on:
            - Food options (e.g., veg, spice, gluten-free)
            - Menu variety
            - Pricing differences
            - Any contrast in services (e.g., ambience, hours)
            
            Chat History:
            {chat_history}
            
                                                          
            Use this data:
            {context}
            
            Question: {question}
            
            Comparison:
            """)
            
            comparison_chain = LLMChain(llm=llm, prompt=compare_prompt)
            
            def compare_agent(question):
                docs = retriever.get_relevant_documents(question)
                if not docs:
                    return fallback_agent()
                ctx = "\n".join([doc.page_content for doc in docs])
                chat_his = get_recent_chat_history(3)
                return comparison_chain.run({"context": ctx, "question": question, "chat_history": chat_his})
            
            # --------------------------------------
            # Diet/Price Filter
            # --------------------------------------
            diet_prompt = PromptTemplate.from_template("""
            You are helping users find restaurants or items based on dietary needs or price.
            
            Chat History:
            {chat_history}
            
            
            Context:
            {context}
            Question: {question}
            
            Answer:
            """)
            
            diet_chain = LLMChain(llm=llm, prompt=diet_prompt)
            
            def diet_agent(question):
                docs = retriever.get_relevant_documents(question)
                if not docs:
                    return fallback_agent()
                ctx = "\n".join([doc.page_content for doc in docs])
                chat_his = get_recent_chat_history(3)
                return diet_chain.run({"context": ctx, "question": question, "chat_history": chat_his})
            
            # --------------------------------------
            # FallBack Agent (General QA)
            # --------------------------------------
            def fallback_agent():
                return (
                    "🤖 I'm here to help you with information about restaurants in Kanpur! "
                    "If you have questions about menu items, prices, comparisons, or dietary options, "
                    "feel free to ask. 😊"
                )
            
            # --------------------------------------
            # Dispatcher (Intent Router)
            # --------------------------------------
            def route_agent(question):
                intent = classify_intent(question)
                st.session_state.last_intent = intent
            
                if intent == "qa":
                    result = qa_chain({"question": question, "chat_history": get_recent_chat_history(3)})
                    answer = result["answer"]
                elif intent == "comparison":
                    answer = compare_agent(question)
                elif intent == "diet":
                    answer = diet_agent(question)
                else:
                    answer = fallback_agent()
            
                memory.save_context({"question": question}, {"answer": answer})
            
                return answer
            
            return route_agent, df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

# Load environment variables
env_loaded = load_env_vars()

# Set up sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot helps you find the perfect restaurant in Kanpur.
    
    **You can ask about:**
    - Restaurant recommendations
    - Specific cuisines
    - Menu items and prices
    - Vegetarian or dietary options
    - Compare two or more restaurants
    - Restaurant contact information
    """)
    
    st.header("Examples")
    example_questions = [
        "Which restaurants serve Chinese food?",
        "What are some good restaurants for vegetarians?",
        "Compare Pizza Hut and Domino's",
        "Which restaurants are open after 10 PM?",
        "Tell me about restaurants with outdoor seating",
    ]
    
    for question in example_questions:
        if st.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    # Show data statistics
    if st.session_state.setup_complete:
        st.header("Dataset Stats")
        st.write(f"Restaurants loaded: {len(st.session_state.df)}")
        
    # Debug section (can be uncommented for debugging)
    # if st.checkbox("Show Debug Info"):
    #     if "last_intent" in st.session_state:
    #         st.write(f"Last detected intent: {st.session_state.last_intent}")

# Main content area - Chat interface
if not st.session_state.setup_complete:
    load_status = st.status("Setting up the restaurant recommendation system...")
    load_status.update(label="Loading restaurant data and initializing models...", state="running")
    
    agent, df = load_data_and_models()
    
    if agent is not None:
        st.session_state.route_agent = agent
        st.session_state.df = df
        st.session_state.setup_complete = True
        load_status.update(label="Setup complete!", state="complete")
        st.rerun()
    else:
        load_status.update(label="Failed to load data. Please check your dataset.", state="error")

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about restaurants in Kanpur..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.route_agent(prompt)
                st.write(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})