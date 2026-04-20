# full_truth_grounder_final.py

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Claim Extraction
# -----------------------------
nlp = spacy.load("en_core_web_sm")

def extract_claims(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# -----------------------------
# Knowledge Base + FAISS
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

knowledge_base = []
documents = []

def create_knowledge_base(docs):
    global knowledge_base, documents
    documents = docs
    knowledge_base = [embedding_model.encode(doc) for doc in docs]

def retrieve_evidence(claim, top_k=2):
    claim_vec = embedding_model.encode(claim)
    index = faiss.IndexFlatL2(len(claim_vec))

    if knowledge_base:
        index.add(np.array(knowledge_base))
        D, I = index.search(np.array([claim_vec]), top_k)
        return [documents[i] for i in I[0]]
    return []

# -----------------------------
# NLI Verification
# -----------------------------
nli_classifier = pipeline(
    "text-classification",
    model="roberta-large-mnli"
)

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

# -----------------------------
# Claim Correction (FINAL FIXED)
# -----------------------------
class EditorAgent:
    def __init__(self):
        print("[INFO] Loading FLAN-T5 model...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def correct_claim(self, claim, evidence):
        prompt = f"""
You are a strict fact-checking AI.

Your task is to CORRECT the claim using the given evidence.

Claim: {claim}
Evidence: {evidence}

Rules:
1. If the claim is incorrect → rewrite it correctly using evidence
2. Do NOT repeat the wrong information
3. Output ONLY one corrected sentence
4. Ensure the corrected fact is accurate

Corrected Sentence:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 🔥 Fallback fix (very important)
        if claim.lower() in result.lower() or len(result) < 10:
            return evidence

        return result

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("🔍 Running Full Truth Grounder...\n")

    # Knowledge Base
    docs = [
        "Barack Obama was born in Hawaii.",
        "The Earth revolves around the Sun once every 365 days.",
        "India gained independence in 1947 from British rule."
    ]

    create_knowledge_base(docs)

    # Input text
    text = "Barack Obama was born in Hawaii. The Earth revolves around the Sun. India became independent in 1950."

    claims = extract_claims(text)
    print(f"📌 Extracted Claims: {claims}\n")

    agent = EditorAgent()

    for claim in claims:
        evidences = retrieve_evidence(claim, top_k=2)

        print(f"🔎 Claim: {claim}")
        print(f"📚 Retrieved Evidence: {evidences}")

        status, label, score, best_evidence = verify_claim(claim, evidences)
        print(f"📊 NLI Verification: {label.upper()} ({score:.2f})")

        if status == "needs_correction":
            corrected = agent.correct_claim(claim, best_evidence)
            print(f"❌ Incorrect → ✅ Corrected Claim: {corrected}\n")
        elif status == "correct":
            print("✅ Claim is correct.\n")
        else:
            print("⚠️ Claim is uncertain.\n")