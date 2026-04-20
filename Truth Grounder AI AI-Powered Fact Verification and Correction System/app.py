import streamlit as st

# ✅ MUST BE FIRST
st.set_page_config(page_title="Truth Grounder AI", layout="centered")

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    nli = pipeline("text-classification", model="roberta-large-mnli")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return nlp, embed_model, nli, tokenizer, model

nlp, embedding_model, nli_classifier, tokenizer, model = load_models()

# -----------------------------
# KNOWLEDGE BASE
# -----------------------------
documents = [
    "Barack Obama was born in Hawaii.",
    "The Earth revolves around the Sun once every 365 days.",
    "India gained independence in 1947 from British rule."
     "The Earth is spherical.",
    "The Sun rises in the east.",
    "Water boils at 100 degrees Celsius.",
    "India is located in Asia.",
    "Humans need oxygen to survive."
     "The Moon orbits the Earth.",
    "Fire is hot and produces heat.",
    "Ice is solid water."
]

knowledge_base = [embedding_model.encode(doc) for doc in documents]

# -----------------------------
# FUNCTIONS
# -----------------------------
def extract_claims(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def retrieve_evidence(claim, top_k=2):
    claim_vec = embedding_model.encode(claim)
    index = faiss.IndexFlatL2(len(claim_vec))
    index.add(np.array(knowledge_base))
    D, I = index.search(np.array([claim_vec]), top_k)
    return [documents[i] for i in I[0]]

def verify_claim(claim, evidences):
    results = []
    for evidence in evidences:
        result = nli_classifier(f"{evidence} </s> {claim}")[0]
        results.append((result, evidence))

    best_result, best_evidence = max(results, key=lambda x: x[0]['score'])

    label = best_result['label'].lower()
    score = best_result['score']

    if "entail" in label:
        status = "correct"
    elif "contradict" in label:
        status = "needs_correction"
    else:
        status = "uncertain"

    return status, label, score, best_evidence

def correct_claim(claim, evidence):
    prompt = f"""
Correct the factual error using the evidence.

Claim: {claim}
Evidence: {evidence}

Return only corrected sentence.
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if claim.lower() in result.lower():
        return evidence
    return result

# -----------------------------
# UI DESIGN (ADVANCED)
# -----------------------------
st.title("🧠 Truth Grounder AI")
st.markdown("### 🔍 AI-Powered Fact Verification System")
st.markdown("---")

user_input = st.text_area("✍️ Enter your text:", height=150)

if st.button("🚀 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("🔄 Processing..."):
            claims = extract_claims(user_input)

            correct_count = 0
            incorrect_count = 0

            for claim in claims:
                st.markdown("## 🔎 Claim")
                st.info(claim)

                evidences = retrieve_evidence(claim)

                st.markdown("### 📚 Evidence")
                for i, ev in enumerate(evidences, 1):
                    st.write(f"{i}. {ev}")

                status, label, score, best_evidence = verify_claim(claim, evidences)

                st.markdown("### 📊 Verification")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Label", label.upper())
                with col2:
                    st.metric("Confidence", f"{score:.2f}")

                st.progress(float(score))

                st.markdown("### 🎯 Decision")

                if status == "correct":
                    st.success("✅ Correct Claim")
                    correct_count += 1

                elif status == "needs_correction":
                    incorrect_count += 1
                    corrected = correct_claim(claim, best_evidence)

                    st.error("❌ Incorrect Claim")
                    st.success(f"✅ Corrected: {corrected}")

                    st.markdown("**📌 Best Evidence Used:**")
                    st.info(best_evidence)

                else:
                    st.warning("⚠️ Uncertain Claim")

                st.markdown("---")

            # -----------------------------
            # SUMMARY SECTION
            # -----------------------------
            st.markdown("## 📊 Summary")
            st.success(f"✅ Correct Claims: {correct_count}")
            st.error(f"❌ Incorrect Claims: {incorrect_count}")