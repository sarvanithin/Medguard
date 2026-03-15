"""
Curated test fixtures with known-good expected outcomes.
All PHI is synthetic — generated values, never real patient data.
"""

# PHI test cases — text + expected detection results
PHI_CASES = [
    {
        "text": "Patient SSN: 123-45-6789 presents with chest pain",
        "expected_entities": ["SSN"],
        "redacted_contains": "[REDACTED]",
        "not_redacted_contains": "chest pain",
    },
    {
        "text": "DOB: 03/15/1980, call (555) 867-5309 for appointment",
        "expected_entities": ["DOB", "PHONE"],
        "redacted_contains": "[REDACTED]",
        "not_redacted_contains": "appointment",
    },
    {
        "text": "Contact patient at john.doe@example.com regarding MRN: 12345678",
        "expected_entities": ["EMAIL", "MRN"],
        "redacted_contains": "[REDACTED]",
        "not_redacted_contains": "regarding",
    },
    {
        "text": "The patient takes aspirin 81mg daily for cardiovascular prevention.",
        "expected_entities": [],
        "not_redacted_contains": "aspirin",
        "note": "No PHI — clinical text should not be modified",
    },
    {
        "text": "Hypertension management with lisinopril is well-documented in guidelines.",
        "expected_entities": [],
        "not_redacted_contains": "lisinopril",
        "note": "No PHI in purely clinical text",
    },
    {
        "text": "Patient: John Smith, DOB: 01/15/1975, SSN 987-65-4321",
        "expected_entities": ["NAME_LABELED", "DOB", "SSN"],
        "redacted_contains": "[REDACTED]",
    },
]

# Drug interaction cases — known good outcomes
DRUG_INTERACTION_CASES = [
    {
        "drugs": ["warfarin", "aspirin"],
        "expected_min_severity": "high",
        "expected_interaction_found": True,
        "note": "Classic anticoagulant + antiplatelet interaction",
    },
    {
        "drugs": ["sertraline", "phenelzine"],
        "expected_min_severity": "contraindicated",
        "expected_interaction_found": True,
        "note": "SSRI + MAOI = serotonin syndrome risk",
    },
    {
        "drugs": ["simvastatin", "clarithromycin"],
        "expected_min_severity": "contraindicated",
        "expected_interaction_found": True,
        "note": "Statin + macrolide CYP3A4 inhibition",
    },
    {
        "drugs": ["metformin", "lisinopril"],
        "expected_min_severity": None,
        "expected_interaction_found": False,
        "note": "Common safe combination for T2DM + hypertension",
    },
    {
        "drugs": ["atorvastatin", "amlodipine"],
        "expected_min_severity": "moderate",
        "expected_interaction_found": True,
        "note": "Mild CYP3A4 interaction, moderate concern",
    },
]

# Scope classification cases
SCOPE_CASES = [
    {
        "text": "What medication should I take for my headache?",
        "expected_in_scope": True,
        "note": "Clear medical question",
    },
    {
        "text": "What are the symptoms of diabetes?",
        "expected_in_scope": True,
        "note": "Medical diagnosis question",
    },
    {
        "text": "Can I sue my doctor for malpractice?",
        "expected_in_scope": False,
        "expected_category": "out_of_scope_legal",
        "note": "Legal question",
    },
    {
        "text": "Will my insurance cover this surgery?",
        "expected_in_scope": False,
        "expected_category": "out_of_scope_financial",
        "note": "Financial/insurance question",
    },
    {
        "text": "I have chest pain and shortness of breath.",
        "expected_in_scope": True,
        "note": "Clinical symptoms — medical scope",
    },
    {
        "text": "What is the lawsuit settlement deadline?",
        "expected_in_scope": False,
        "expected_category": "out_of_scope_legal",
    },
]

# Hallucination test cases
HALLUCINATION_CASES = [
    {
        "text": "Take draxilomycin 500mg twice daily for your infection.",
        "expected_flag_types": ["fake_drug_name"],
        "note": "Fabricated drug name ending in -mycin",
    },
    {
        "text": "The recommended dose of acetaminophen is 20,000mg per day.",
        "expected_flag_types": ["impossible_dosage"],
        "note": "Lethal acetaminophen dose (max is 4000mg/day)",
    },
    {
        "text": "Take ibuprofen 400mg three times daily with food.",
        "expected_flag_types": [],
        "note": "Valid dosage within safe range",
    },
    {
        "text": "This medication will definitely cure your condition with absolutely no side effects.",
        "expected_flag_types": ["confident_unsupported_claim"],
        "note": "Overconfident language",
    },
    {
        "text": "Metformin 500mg is typically started twice daily for type 2 diabetes.",
        "expected_flag_types": [],
        "note": "Standard clinical recommendation — no flags",
    },
]

# Synthetic PHI values for property-based testing (not real data)
SYNTHETIC_PHI = {
    "ssns": [
        "123-45-6789", "987-65-4321", "555-12-3456", "001-23-4567",
    ],
    "phones": [
        "(555) 867-5309", "555-555-1234", "+1 800 555 0100", "555.555.5555",
    ],
    "emails": [
        "patient@example.com", "john.doe@hospital.org", "test+filter@domain.co",
    ],
    "dobs": [
        "01/15/1980", "12/31/1955", "03-05-2001", "July 4, 1975",
    ],
    "mrns": [
        "MRN: 1234567", "MRN#89012345", "mrn 9876543",
    ],
}
