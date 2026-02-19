import re

def extract_resume_fields(text: str):
    fields = {}

    fields["email"] = _regex(text, r"[\w\.-]+@[\w\.-]+\.\w+")
    fields["phone"] = _regex(text, r"\+?\d[\d\s\-]{8,}")
    fields["education"] = _contains_any(
        text, ["bachelor", "master", "degree", "university"]
    )

    return fields

def _regex(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return {"value": match.group(), "confidence": 0.9}
    return {"value": None, "confidence": 0.0}

def _contains_any(text, keywords):
    for k in keywords:
        if k in text.lower():
            return {"value": k, "confidence": 0.6}
    return {"value": None, "confidence": 0.0}
