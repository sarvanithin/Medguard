"""
medguard basic usage examples.

Run: python examples/basic_usage.py
"""
import asyncio
from medguard import MedGuard


def example_phi_check():
    print("\n=== PHI Detection Example ===")
    mg = MedGuard()

    text = "Patient John Smith, SSN: 123-45-6789, DOB: 01/15/1980 asks about metformin."
    result = mg.check(text)

    print(f"PHI detected: {result.phi_result.phi_detected}")
    if result.phi_result.phi_detected:
        for match in result.phi_result.matches:
            print(f"  - {match.entity_type}: '{match.text}'")
        print(f"Redacted: {result.phi_result.processed}")


def example_scope_check():
    print("\n=== Scope Enforcement Example ===")
    mg = MedGuard()

    queries = [
        "What are the side effects of metformin?",
        "Can I sue my doctor for malpractice?",
        "Will my insurance cover this surgery?",
        "What is the maximum dose of ibuprofen?",
    ]

    for query in queries:
        result = mg.check(query)
        scope = result.scope_result
        if scope:
            status = "IN SCOPE" if scope.in_scope else f"OUT OF SCOPE ({scope.category.value})"
            print(f"  [{status}] {query[:60]}")


def example_drug_safety():
    print("\n=== Drug Safety Check Example ===")
    mg = MedGuard()

    texts = [
        "Patient is taking warfarin and aspirin daily.",
        "Sertraline and phenelzine prescribed together.",
        "Patient takes metformin 500mg and lisinopril 10mg.",
    ]

    for text in texts:
        result = mg.check(text)
        drug = result.drug_result
        if drug:
            if drug.interactions:
                for interaction in drug.interactions:
                    print(
                        f"  INTERACTION [{interaction.severity.value.upper()}]: "
                        f"{interaction.drug_a} + {interaction.drug_b}"
                    )
            else:
                print(f"  No interactions found for: {text[:50]}")


async def example_chat():
    print("\n=== Guardrailed Chat Example ===")
    mg = MedGuard()

    questions = [
        "What are the common side effects of metformin?",
        "My SSN is 123-45-6789 and I take warfarin. Should I also take aspirin?",
    ]

    for question in questions:
        print(f"\nQuestion: {question[:80]}")
        response = await mg.achat(question)
        print(f"Response: {response[:200]}...")


if __name__ == "__main__":
    example_phi_check()
    example_scope_check()
    example_drug_safety()
    # asyncio.run(example_chat())  # Requires API key
    print("\nDone. Start the server with: python -m medguard")
