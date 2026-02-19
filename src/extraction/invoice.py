import re


def extract_invoice_fields(text: str) -> dict:
    fields = {}

    fields["invoice_number"] = _regex(text, r"invoice\s*(?:no\.?|number|#)[:\s]*([A-Z0-9\-]+)", group=1)
    fields["date"] = _regex(text, r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b", group=1)
    fields["total_amount"] = _regex(text, r"(?:total|amount\s*due|grand\s*total)[:\s\$]*([0-9,]+\.\d{2})", group=1)
    fields["vendor"] = _first_line(text)
    fields["currency"] = _regex(text, r"\b(USD|GBP|EUR|INR|CAD|AUD)\b", group=1)

    return fields


def _regex(text: str, pattern: str, group: int = 0) -> dict:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return {"value": match.group(group), "confidence": 0.85}
    return {"value": None, "confidence": 0.0}


def _first_line(text: str) -> dict:
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 3:
            return {"value": line, "confidence": 0.5}
    return {"value": None, "confidence": 0.0}
