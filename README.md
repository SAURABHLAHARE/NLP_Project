# 🧠 Truth Grounder AI

### AI-Powered Fact Verification and Correction System

---

## 📌 Overview

Truth Grounder AI is an advanced Natural Language Processing (NLP) system designed to automatically verify factual claims and provide corrected outputs.
The system combines multiple AI techniques such as claim extraction, retrieval-augmented generation (RAG), and natural language inference (NLI) to ensure high-quality fact verification.

This project demonstrates how modern AI pipelines can be used to combat misinformation and improve content reliability.

---

## 🚀 Key Features

* 🔍 **Claim Extraction** – Identifies factual claims from raw text
* 📚 **RAG-based Retrieval** – Fetches relevant supporting information
* 🧠 **NLI Verification** – Determines whether a claim is true or false
* ✍️ **Automated Correction** – Rewrites incorrect claims with accurate information
* ⚡ **Modular Design** – Easy to extend and improve

---

## 🏗️ System Architecture

```
User Input Text
        ↓
Claim Extractor
        ↓
RAG Retriever
        ↓
NLI Verifier
        ↓
Editor Agent
        ↓
Final Verified & Corrected Output
```

---

## 📂 Project Structure

```
NLP_Project/
│
├── claim_extractor.py       # Extracts claims from input text  
├── rag_retriever.py         # Retrieves relevant evidence  
├── nli_verifier.py          # Verifies claims using NLI models  
├── editor_agent.py          # Corrects or rewrites claims  
├── full_truth_grounder.py   # Main pipeline controller  
├── phase1_setup.py          # Initial setup and configuration  
├── requirements.txt         # Project dependencies  
└── README.md                # Project documentation  
```

---

## 🛠️ Tech Stack

* **Programming Language:** Python 🐍
* **Core Concepts:** NLP, Machine Learning
* **Techniques:**

  * Retrieval-Augmented Generation (RAG)
  * Natural Language Inference (NLI)
  * Transformer-based models

---

## ⚙️ Installation Guide

### Step 1: Clone Repository

```bash
git clone https://github.com/Sai0045/NLP_Project.git
cd NLP_Project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python full_truth_grounder.py
```

---

## 📊 Example

### Input:

```
Vaccines cause autism.
```

### Output:

* Claim extracted
* Evidence retrieved
* Verification result: ❌ False
* Corrected statement generated

---

## 🎯 Use Cases

* Fake News Detection
* Automated Fact Checking
* AI Assistants
* Research Validation
* Content Moderation

---

## 📈 Future Enhancements

* 🌐 Web Interface using Streamlit
* 🔗 Integration with live APIs
* 🌍 Multilingual support
* 📊 Improved model accuracy with fine-tuning

---

## 👨‍💻 Author

**Sairaj Abhale**
AI & ML Student | Aspiring AI Engineer

---

## 🤝 Contribution

Contributions are welcome!
Feel free to fork this repository and submit pull requests.

---
