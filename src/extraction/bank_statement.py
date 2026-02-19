import re


def extract_bank_statement_fields(text: str) -> dict:
    fields = {}

    fields["account_number"] = _regex(text, r"a[/]?c[:\s#]*([A-Z0-9]{6,20})", group=1)
    fields["bank_name"] = _first_line(text)
    fields["statement_period"] = _regex(
        text,
        r"(?:between|period|statement)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s*(?:and|to|-)\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        group=0,
    )
    fields["currency"] = _regex(text, r"\b(USD|GBP|EUR|INR|CAD|AUD)\b", group=1)
    fields["opening_balance"] = _regex(
        text, r"(?:opening|beginning)\s*balance[:\s\$]*([0-9,]+\.\d{2})", group=1
    )
    fields["closing_balance"] = _regex(
        text, r"(?:closing|ending)\s*balance[:\s\$]*([0-9,]+\.\d{2})", group=1
    )

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
