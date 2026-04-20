import spacy

class ClaimExtractor:
    def __init__(self):
        print("[INFO] Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")

    def is_claim(self, sentence):
        """
        Keep sentences that likely contain factual claims:
        - Has named entities (person, place, date, etc.)
        - OR contains numbers
        """
        doc = self.nlp(sentence)

        has_entity = len(doc.ents) > 0
        has_number = any(token.like_num for token in doc)

        return has_entity or has_number

    def extract_claims(self, text):
        doc = self.nlp(text)

        claims = []
        for sent in doc.sents:
            sentence = sent.text.strip()

            if self.is_claim(sentence):
                claims.append(sentence)

        return claims


if __name__ == "__main__":
    print("🔍 Running Claim Extraction Test...\n")

    text = """
    Barack Obama was born in Hawaii. 
    He loves playing basketball. 
    The Earth revolves around the Sun. 
    Python is a programming language.
    India became independent in 1947.
    """

    extractor = ClaimExtractor()
    claims = extractor.extract_claims(text)

    print("📌 Extracted Claims:\n")
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim}")