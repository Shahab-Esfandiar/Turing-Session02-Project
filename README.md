# 🧠 Advanced RAG Architecture: Agentic Memory & Matryoshka Benchmarking

An enterprise-grade AI engineering repository focused on solving two critical challenges in modern RAG (Retrieval-Augmented Generation) architectures: **Long-Term Memory Consistency (Conflict Resolution)** and **Vector Dimensionality Optimization (Matryoshka Representation Learning)**.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 🎯 Project Overview

This repository contains two primary Jupyter Notebooks that implement state-of-the-art solutions for AI agent memory and retrieval efficiency:
1. **`Session02-Project_0104.ipynb`**: Implements an **Agentic Memory Manager** using an LLM-in-the-loop to prevent vector database pollution and maintain logical consistency. Includes a formal **ADR (Architecture Decision Record)**.
2. **`Session02-Project_0203.ipynb`**: Conducts a massive **10,000-item Stress Test** to benchmark the exact breaking points of Matryoshka embeddings across various dimensions and complex datasets.

---

## ✨ Part 1: Agentic Memory Consistency (ChromaDB + LLM Judge)

### The Problem
Standard Vector Databases (VDBs) only understand mathematical semantic similarity, not logical contradictions or temporal state updates. Relying purely on similarity thresholds leads to either **Data Loss** (overwriting valid complementary facts) or **Memory Pollution** (appending conflicting facts).

### The Solution
Implemented a highly intelligent, four-tier processing architecture:
1. **Semantic Ingestion:** Generates embeddings for incoming facts and retrieves top candidates from **ChromaDB**.
2. **Deterministic LLM Arbitration:** Uses `gpt-4o` with `temperature=0.0` as an isolated Memory Judge.
3. **Cognitive Resolution:** The LLM detects logical contradictions. If the facts co-exist logically, it returns `"NEW"`. If it's a temporal update (e.g., changed address), it returns the exact `UUID` of the obsolete record.
4. **Atomic Database Execution:** Executes an atomic `upsert` to cleanly erase contradictions or appends new records.

> 📄 **Included:** A comprehensive **ADR-01** detailing the decision, rejected alternatives (like Time-Weighted Decay and Store-All), and trade-offs.

---

## 📊 Part 2: Matryoshka Embeddings & Massive Stress Testing

### The Experiment
To rigorously evaluate the limits of Matryoshka Representation Learning (MRL), we scaled the dataset in three phases to simulate a dense production environment:
1. **Baseline Test:** 30 highly distinct items.
2. **Standard Stress Test:** 1,000 items (tech profiles + semantic noise).
3. **Massive Stress Test:** 10,000 highly dense vectors (6,000 extremely similar profiles + 4,000 distraction paragraphs).

### Dimensional Truncation & Metrics
We sliced the `text-embedding-3-small` embeddings from **1536 dimensions** down to **512, 128, 64, and 32 dimensions** and evaluated them using industry-standard search metrics:
* **Recall@5 & Recall@10**
* **mAP (Mean Average Precision)**
* **NDCG (Normalized Discounted Cumulative Gain)**
* **Latency (µs)** and **Memory Footprint (Bytes)**

### Key Findings & Insights
* 🟢 **The "Sweet Spot" (128 Dimensions):** Even in the 10,000-item test, 128 dimensions maintained a flawless `1.00` score across all precision metrics while reducing the RAM footprint by **91.6%** (from 122MB to 10MB) and accelerating queries dramatically.
* 🔴 **The Breaking Point (32 Dimensions):** Proved experimentally that extreme truncation in a highly dense dataset causes severe *Vector Collision*. At 32 dimensions, the model could no longer differentiate subtle nuances, dropping `Recall@5` to 0.60 and `mAP@5` to 0.28.

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
```

**2. Install dependencies**
```bash
pip install openai chromadb scikit-learn pandas numpy python-dotenv httpx
```

**3. Configure Environment Variables**
Create a .env file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=[https://api.avalai.ir/v1](https://api.avalai.ir/v1)  # Or standard OpenAI URL
```

**4. Execute the Notebooks**
Launch Jupyter Notebook and run the cells in Session02-Project_0104.ipynb and Session02-Project_0203.ipynb sequentially to observe the Agentic Memory Manager and the Benchmarking processes.

***Note:*** *Generating embeddings for the 10,000-item stress test uses API chunking and may take a few minutes to complete depending on rate limits.*

---

## 🛠️ Tech Stack & Libraries

* **Core:** `Python`, `pandas`, `numpy`, `itertools`
* **AI & Machine Learning:** `openai` (`gpt-4o`, `text-embedding-3-small`), `scikit-learn` (Cosine Similarity)
* **Vector Database:** `chromadb` (Persistent Local Storage)
* **Network & Environment:** `httpx`, `python-dotenv`
