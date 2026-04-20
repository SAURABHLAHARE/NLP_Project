from transformers import pipeline


class NLIVerifier:
    def __init__(self):
        print("[INFO] Loading NLI model...")
        self.classifier = pipeline(
            "text-classification",
            model="typeform/distilbert-base-uncased-mnli",
            return_all_scores=True
        )

    def verify(self, claim, evidence):
        """
        Proper NLI: Evidence = premise, Claim = hypothesis
        """
        sequence = f"{evidence} </s> {claim}"
        result = self.classifier(sequence)
        return result


if __name__ == "__main__":
    print("🔍 Running NLI Verification Test...\n")

    verifier = NLIVerifier()

    claim = "India became independent in 1947."

    evidence_list = [
        "India gained independence in 1947 from British rule.",
        "India became independent in 1950.",
        "Python is a programming language."
    ]

    for evidence in evidence_list:
        result = verifier.verify(claim, evidence)

        print(f"\n🧾 Claim: {claim}")
        print(f"📚 Evidence: {evidence}")
        print("📊 Result:", result)