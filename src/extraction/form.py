import re


def extract_form_fields(text: str) -> dict:
    fields = {}

    fields["form_number"] = _regex(text, r"form\s*(?:no\.?|number|#)?[:\s]*([A-Z0-9\-]{4,20})", group=1)
    fields["date"] = _regex(text, r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b", group=1)
    fields["name"] = _regex(text, r"(?:name|applicant|claimant)[:\s]+([A-Za-z\s]{3,50})", group=1)
    fields["id_number"] = _regex(text, r"(?:id|ssn|tax\s*id|ein)[:\s#]*([A-Z0-9\-]{5,20})", group=1)

    return fields


def _regex(text: str, pattern: str, group: int = 0) -> dict:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return {"value": match.group(group).strip(), "confidence": 0.8}
    return {"value": None, "confidence": 0.0}
