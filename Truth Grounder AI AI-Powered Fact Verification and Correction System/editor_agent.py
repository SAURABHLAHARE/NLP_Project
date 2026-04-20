from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EditorAgent:
    def __init__(self):
        print("[INFO] Loading FLAN-T5 model...")

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def correct_claim(self, claim, evidence):
        prompt = f"""
You are a fact-checking system.

Example 1:
Claim: The Earth is flat.
Evidence: The Earth is spherical.
Answer: The Earth is spherical.

Example 2:
Claim: India became independent in 1950.
Evidence: India gained independence in 1947 from British rule.
Answer: India became independent in 1947.

Now:

Claim: {claim}
Evidence: {evidence}
Answer:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return result.strip()


if __name__ == "__main__":
    print("🔍 Running Correction Agent Test...\n")

    agent = EditorAgent()

    claim = "India became independent in 1950."
    evidence = "India gained independence in 1947 from British rule."

    corrected = agent.correct_claim(claim, evidence)

    print("❌ Original Claim:", claim)
    print("📚 Evidence:", evidence)
    print("✅ Corrected:", corrected)