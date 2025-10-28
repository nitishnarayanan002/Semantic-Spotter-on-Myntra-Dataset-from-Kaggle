

# 🛍️ Myntra Product Semantic Search using Sentence Transformers & ChromaDB

This project demonstrates how to build a **semantic search engine** for Myntra’s product catalog using **Sentence Transformers**, **ChromaDB**, and **LLM-based query response (Perplexity API)**.
It enables natural language search queries (like *“show me cotton shirts under 999 from Roadster”*) and retrieves the most contextually relevant products — going beyond traditional keyword matching.

---

## 🚀 **Project Overview**

Traditional search systems rely on keyword matching, which often fails to capture the intent or meaning behind user queries.
This project introduces a **semantic search pipeline** that:

* Converts product data into **dense embeddings**
* Stores them in a **vector database (ChromaDB)**
* Retrieves contextually relevant products using **cosine similarity**
* Optionally uses **LLMs (Perplexity / GPT)** to generate conversational responses

---

## 🧠 **Architecture**

```plaintext
Myntra Product Dataset
        ↓
Text Preprocessing & Cleaning
        ↓
Embedding Generation (SentenceTransformer)
        ↓
Vector Storage (ChromaDB)
        ↓
Semantic Retrieval (Similarity Search)
        ↓
LLM-based Response (Perplexity / GPT)
```

---

## 📦 **Tech Stack**

| Component           | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| **Language**        | Python                                                         |
| **Environment**     | Google Colab                                                   |
| **Libraries**       | pandas, numpy, sentence-transformers, chromadb, langchain-core |
| **Embeddings**      | all-MiniLM-L6-v2                                               |
| **Vector DB**       | ChromaDB                                                       |
| **LLM Integration** | Perplexity API                                                 |
| **Visualization**   | Matplotlib (optional)                                          |

---

## ⚙️ **Setup Instructions**

###  Install Dependencies

```bash
!pip install -q pandas numpy sentence-transformers chromadb langchain openai
```

###  Load Dataset

Upload or link your Myntra dataset in Colab:

```python
df = pd.read_csv("/content/Myntra-Ecommerce__20231101_20231130_sample.csv")
```

###  Generate Embeddings

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
df['embeddings'] = df['text'].apply(lambda x: model.encode(x))
```

### Store in ChromaDB

```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("myntra_products")
```

### Query Semantically

```python
query = "Find cotton shirts under 999 from brand US Polo Assn."
# Retrieve semantically similar results
results = collection.query(query_texts=[query], n_results=5)
```

---

## 💬 **Example Queries**

| Query                                                   | Description                           |
| ---------------------------------------------------     | ------------------------------------- |
| “Find cotton shirts under 999 from brand US Polo Assn.” | Retrieve budget-friendly shirts       |
| “Show red dresses suitable for party.”                  | Fetch occasion-specific clothing      |
| “List running shoes for men under 2000.”                | Filter sportswear by price and gender |

---

## 🔮 **Future Enhancements**

* Add multilingual query support
* Deploy REST API endpoint for search
* Integrate UI using Streamlit or Gradio
* Experiment with fine-tuned domain embeddings

---

## ✨ **Author**

**Nitish Narayanan**


---

Would you like me to format it with badges (e.g., Python version, License, Colab link) to make it look more like a polished open-source README?
