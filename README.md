# 🍽️ Restaurant Data Scraper & RAG-based Chatbot

This project is focused on building a restaurant-aware AI chatbot for **Kanpur**. It integrates **web scraping**, **OCR**, **RAG-based LLM inference**, **intent classification**, and a sleek **Streamlit UI** to answer natural language queries about food and restaurants using real-time menu data from **Zomato** and **Swiggy**.

---

## 🚀 Features

- 📄 Scrapes restaurant data from **Zomato** and **Swiggy** (for Kanpur) using **Selenium**
- 🧾 Extracts text from menu images using **OCR (Tesseract)**
- 🧠 Implements a **RAG (Retrieval-Augmented Generation)** chatbot using:
  - **FAISS** vector store
  - **Hugging Face** embeddings
  - **Groq's LLaMA3-8B** model
- 💬 Supports **intent classification** for:
  - Restaurant QA
  - Comparison
  - Dietary preferences
  - Fallback queries
- 🧪 Experimented in `rag.ipynb`, final app in `rag.py`
- 🧑‍💻 Built with **Streamlit** and deployed
- 🔀 Logic routing based on detected user intent

---

## 🧠 Architecture & Workflow


```mermaid
graph TD
    A[User Query] --> B[Intent Classification]
    B --> C1[QA Chain]
    B --> C2[Comparison Chain]
    B --> C3[Diet Chain]
    B --> C4[Fallback Chain]
    
    C1 --> D[Query FAISS Vector DB]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E[Retrieve Menu Chunks]
    E --> F[LLM (Groq LLaMA3)]
    F --> G[Streamlit UI]

```

## 🛠 Technologies Used

| Task                        | Tool/Library                       |
|-----------------------------|------------------------------------|
| Web Scraping                | Selenium                           |
| OCR                         | Tesseract, `pytesseract`           |
| Embeddings                  | `HuggingFaceEmbeddings`            |
| Vector DB                   | FAISS                              |
| LLM Inference               | LangChain, Groq, LLaMA3-8B         |
| App Framework               | Streamlit                          |
| Prompt Engineering & RAG   | LangChain                          |
| CSV Parsing & Merging      | `pandas`                           |

markdown
Copy code
# 🍽️ Zomato RAG Chatbot

A conversational assistant to help you find restaurants, compare menus, and get dietary suggestions — powered by **LangChain**, **Streamlit**, and **Groq's LLaMA3-8B**.

---

## 🗂 Project Structure

📁 your-repo/
├── merged.csv # Combined restaurant data with menu text
├── rag.py # Final Streamlit app
├── rag.ipynb # Development notebook
├── .env # Optional Groq API keys
├── requirements.txt
└── README.md


---

## 💡 Intent Routing Logic

The chatbot handles 4 types of queries using keyword-based classification:

- **qa**  
  _Examples_:  
  - “What are some Chinese restaurants?”  
  - “Show me options near IIT”

- **comparison**  
  _Examples_:  
  - “Compare KFC and Burger King”  
  - “Which is cheaper?”

- **diet**  
  _Examples_:  
  - “Suggest vegan dishes”  
  - “I want low-calorie meals”

- **fallback**  
  For unclear or out-of-scope queries.

> Each query type is routed to a dedicated LangChain pipeline using custom prompt templates.

---

## 🖥️ Running the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/zomato-rag-chatbot.git
cd zomato-rag-chatbot
2. Install Requirements
bash
Copy code
pip install -r requirements.txt
3. Setup Environment
Add your GROQ API key to a .env file:

bash
Copy code
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
Install Tesseract OCR (for extracting text from menus):

bash
Copy code
# Linux
sudo apt install tesseract-ocr

# macOS (Homebrew)
brew install tesseract
4. Run the App
bash
Copy code
streamlit run rag.py
The app will open in your browser at http://localhost:8501

✨ Features
💬 Natural language chat interface
📋 Real-time menu data retrieval using OCR
⚖️ Restaurant comparison tool
🥗 Smart dietary recommendations (vegan, low-calorie, etc.)
⚡ Fast, efficient responses using RAG + LLaMA3-8B
🖼 App Screenshot
<!-- You can add a screenshot image here --> <!-- ![Screenshot](screenshot.png) -->
🤝 Credits
Built with ❤️ by Shivang

