# Strategic Medical GPQA: Questions where self-consistency provides genuine advantage
# These are ambiguous or require reasoning where voting filters errors
REASONING_TASKS = [
    # Ambiguous/Tricky Medical Questions (Where baseline models often disagree with themselves)
    {
        "name": "GPQA: Medication Interaction",
        "prompt": "A patient takes warfarin and wants to start a new supplement. Which supplement has the HIGHEST risk of interaction? A: Vitamin D, B: Vitamin E, C: Calcium, D: Vitamin B12. Respond with 'Final Answer: <label>'.",
        "answer": "B"  # Vitamin E increases bleeding risk - tricky because multiple vitamins interact
    },
    {
        "name": "GPQA: Differential Diagnosis",
        "prompt": "A 60-year-old presents with fatigue, weight loss, and night sweats. Which diagnosis should be ruled out FIRST? A: Hypothyroidism, B: Depression, C: Tuberculosis or malignancy, D: Anemia. Respond with 'Final Answer: <label>'.",
        "answer": "C"  # Red flag symptoms require urgent evaluation
    },
    {
        "name": "GPQA: Drug Dosing",
        "prompt": "For a patient with renal impairment (GFR 25), which medication requires the MOST significant dose adjustment? A: Lisinopril, B: Metformin, C: Amoxicillin, D: Atorvastatin. Respond with 'Final Answer: <label>'.",
        "answer": "B"  # Metformin is contraindicated - tricky distinction
    },
    {
        "name": "GPQA: Contraindication",
        "prompt": "Which condition is an ABSOLUTE contraindication for starting oral contraceptives? A: Migraines without aura, B: Previous DVT/PE, C: Family history of breast cancer, D: Obesity. Respond with 'Final Answer: <label>'.",
        "answer": "B"  # Absolute vs relative contraindications are subtle
    },
    {
        "name": "GPQA: Pediatric Emergency",
        "prompt": "A 2-year-old has high fever and refuses to move their neck. What is the MOST urgent concern? A: Viral meningitis, B: Bacterial meningitis, C: Torticollis, D: Ear infection. Respond with 'Final Answer: <label>'.",
        "answer": "B"  # Life-threatening vs benign - critical distinction
    },
    {
        "name": "GPQA: Antibiotic Selection",
        "prompt": "For community-acquired pneumonia in a penicillin-allergic patient, which is the BEST first-line choice? A: Cephalexin, B: Azithromycin, C: Amoxicillin, D: Penicillin VK. Respond with 'Final Answer: <label>'.",
        "answer": "B"  # Cross-reactivity makes this tricky
    },
    {
        "name": "GPQA: Lab Interpretation",
        "prompt": "A patient has elevated alkaline phosphatase but normal ALT/AST. Which organ system is MOST likely affected? A: Liver, B: Bone, C: Kidney, D: Heart. Respond with 'Final Answer: <label>'.",
        "answer": "B"  # Isolated alk phos elevation pattern
    },
    {
        "name": "GPQA: Vaccine Timing",
        "prompt": "A patient received the flu vaccine yesterday. When is the EARLIEST they can receive the COVID-19 vaccine? A: Same day was fine, B: Wait 2 weeks, C: Wait 1 month, D: Wait 3 months. Respond with 'Final Answer: <label>'.",
        "answer": "A"  # Guidelines changed - models may have outdated info
    },
    {
        "name": "GPQA: Chest Pain Triage",
        "prompt": "A 35-year-old female with chest pain that worsens with deep breathing. EKG normal. What is the MOST likely diagnosis? A: Myocardial infarction, B: Pulmonary embolism, C: Pleurisy/costochondritis, D: GERD. Respond with 'Final Answer: <label>'.",
        "answer": "C"  # Pleuritic pain characteristics - age/EKG mislead
    },
    {
        "name": "GPQA: Diabetic Ketoacidosis",
        "prompt": "In DKA, which electrolyte abnormality requires IMMEDIATE correction before insulin therapy? A: Hyperglycemia, B: Hypokalemia, C: Metabolic acidosis, D: Hypernatremia. Respond with 'Final Answer: <label>'.",
        "answer": "B"  # Critical sequencing - insulin without K+ is dangerous
    }
]
