import re


def extract_utility_fields(text: str) -> dict:
    fields = {}

    fields["account_number"] = _regex(text, r"(?:account|acct|service\s*address)[:\s#]*([A-Z0-9\-]{5,20})", group=1)
    fields["due_date"] = _regex(text, r"(?:due\s*date|pay\s*by)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", group=1)
    fields["total_due"] = _regex(
        text, r"(?:total\s*amount\s*due|amount\s*due|total\s*due)[:\s\$]*([0-9,]+\.\d{2})", group=1
    )
    fields["service_period"] = _regex(
        text,
        r"(?:service\s*period|billing\s*period)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s*(?:to|-)\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        group=0,
    )
    fields["utility_type"] = _detect_utility_type(text)

    return fields


def _detect_utility_type(text: str) -> dict:
    text_lower = text.lower()
    for utility in ["electricity", "electric", "water", "gas", "sewer", "refuse", "internet", "phone"]:
        if utility in text_lower:
            return {"value": utility, "confidence": 0.75}
    return {"value": None, "confidence": 0.0}


def _regex(text: str, pattern: str, group: int = 0) -> dict:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return {"value": match.group(group).strip(), "confidence": 0.85}
    return {"value": None, "confidence": 0.0}
