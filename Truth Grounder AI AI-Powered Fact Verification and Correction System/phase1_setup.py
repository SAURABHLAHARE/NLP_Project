from transformers import pipeline
from sentence_transformers import SentenceTransformer
import spacy

def test_spacy():
    print("\n[1] Testing spaCy...")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Barack Obama was born in Hawaii.")
    
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")

def test_transformers():
    print("\n[2] Testing Hugging Face Transformer (NLI)...")
    classifier = pipeline("text-classification", model="typeform/distilbert-base-uncased-mnli")
    
    result = classifier("Barack Obama was born in Kenya.", 
                        candidate_labels=["true", "false"])
    
    print("NLI Output:", result)

def test_embeddings():
    print("\n[3] Testing Sentence Transformers...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    sentences = ["The sky is blue.", "The sun is bright."]
    embeddings = model.encode(sentences)
    
    print("Embedding shape:", embeddings.shape)

if __name__ == "__main__":
    print("🔍 Running Phase 1 Setup Tests...")
    
    test_spacy()
    test_transformers()
    test_embeddings()
    
    print("\n✅ Phase 1 Complete: Environment Ready!")
   