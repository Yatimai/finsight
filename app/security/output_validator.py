"""
Output validation: checks response quality and detects anomalies.
Lightweight security layer — monitoring only, no hard blocks.
"""

import re


def validate_response(answer: str) -> dict:
    """
    Validate a generated response.

    Checks:
    - Presence of at least one [Page X] citation
    - Response is not empty
    - Response doesn't mention system prompt or instructions

    Returns:
        Dict with "valid": bool, "anomalies": list[str]
    """
    anomalies = []

    # Check for citations
    citations = re.findall(r"\[Page\s+\d+\]", answer)
    if not citations and "n'apparaît pas" not in answer.lower() and "pas dans les documents" not in answer.lower():
        anomalies.append("no_citations")

    # Check for empty response
    if len(answer.strip()) < 10:
        anomalies.append("empty_response")

    # Check for system prompt leakage indicators
    leakage_indicators = [
        "system prompt",
        "mes instructions",
        "je suis programmé",
        "mon prompt",
        "RÈGLES :",
        "FORMAT :",
        "EXEMPLES :",
        "cache_control",
    ]
    answer_lower = answer.lower()
    for indicator in leakage_indicators:
        if indicator.lower() in answer_lower:
            anomalies.append(f"possible_leakage:{indicator}")

    return {
        "valid": len(anomalies) == 0,
        "anomalies": anomalies,
        "citation_count": len(citations),
    }
