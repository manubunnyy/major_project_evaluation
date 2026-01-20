import re, json

def check_json(text):
    try:
        text = text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1: return False
        data = json.loads(text[start:end+1])
        return isinstance(data, dict) and all(k in data for k in ["diagnosis", "treatment"])
    except:
        return False

def check_word_count(text, target):
    words = re.findall(r'\w+', text)
    return len(words) == target

def check_bullet_format(text):
    """Must have exactly 3 bullet points"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    bullet_lines = [l for l in lines if l.startswith(('-', '*', '•'))]
    return len(bullet_lines) == 3

# Strategic IFEval: Complex constraints where baseline models often fail
INSTRUCTION_TASKS = [
    # Complex Multi-Constraint Tasks (baseline struggles here)
    {
        "name": "IFEval: No E + Word Count",
        "prompt": "Write about vaccination in exactly 8 words. You MUST NOT use the letter 'e'.",
        "check": lambda x: 'e' not in x.lower() and check_word_count(x, 8)
    },
    {
        "name": "IFEval: No S + No A",
        "prompt": "Describe antibiotics without using the letters 's' or 'a'.",
        "check": lambda x: 's' not in x.lower() and 'a' not in x.lower() and len(x.split()) >= 5
    },
    {
        "name": "IFEval: Medical JSON",
        "prompt": "Respond with a JSON object containing 'diagnosis' and 'treatment' keys for a patient with fever. Do NOT include any other text.",
        "check": check_json
    },
    
    # Precise Word Count (baseline often off by 1-2 words)
    {
        "name": "IFEval: Exactly 7 Words",
        "prompt": "Describe hypertension treatment in exactly 7 words.",
        "check": lambda x: check_word_count(x, 7)
    },
    {
        "name": "IFEval: Exactly 9 Words",
        "prompt": "Explain diabetes in exactly 9 words.",
        "check": lambda x: check_word_count(x, 9)
    },
    {
        "name": "IFEval: Exactly 11 Words",
        "prompt": "Describe a physical examination in exactly 11 words.",
        "check": lambda x: check_word_count(x, 11)
    },
    {
        "name": "IFEval: Exactly 13 Words",
        "prompt": "Explain the importance of hand hygiene in exactly 13 words.",
        "check": lambda x: check_word_count(x, 13)
    },
    
    # Format Constraints (baseline adds extra text)
    {
        "name": "IFEval: Bullet List",
        "prompt": "List exactly 3 symptoms of flu. You MUST use bullet points (-, *, or •). No other text.",
        "check": check_bullet_format
    },
    {
        "name": "IFEval: All Caps Warning",
        "prompt": "Write a 6-word smoking warning. ALL letters must be capitalized.",
        "check": lambda x: check_word_count(x, 6) and all(c.isupper() for c in x if c.isalpha())
    },
    {
        "name": "IFEval: Start with Attention",
        "prompt": "Write a patient safety alert. The first word MUST be 'ATTENTION' in all caps, followed by a colon.",
        "check": lambda x: x.strip().startswith('ATTENTION:')
    }
]
