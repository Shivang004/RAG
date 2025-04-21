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
