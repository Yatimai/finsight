"""Tests for output validation and anomaly detection."""

from app.security.output_validator import validate_response


class TestValidateResponse:
    def test_valid_response_with_citations(self):
        answer = "Le CA 2023 est de 86,2 milliards [Page 12]."
        result = validate_response(answer)
        assert result["valid"] is True
        assert result["anomalies"] == []
        assert result["citation_count"] == 1

    def test_multiple_citations(self):
        answer = "Le CA [Page 12] est en hausse [Page 14] par rapport à 2022 [Page 8]."
        result = validate_response(answer)
        assert result["valid"] is True
        assert result["citation_count"] == 3

    def test_no_citations_flags_anomaly(self):
        answer = "Le chiffre d'affaires est de 86,2 milliards d'euros."
        result = validate_response(answer)
        assert result["valid"] is False
        assert "no_citations" in result["anomalies"]

    def test_abstention_response_is_valid_without_citations(self):
        answer = "Cette information n'apparaît pas dans les documents fournis."
        result = validate_response(answer)
        assert result["valid"] is True
        assert "no_citations" not in result["anomalies"]

    def test_abstention_variant(self):
        answer = "Ces données ne sont pas dans les documents consultés."
        result = validate_response(answer)
        assert result["valid"] is True

    def test_empty_response(self):
        answer = "   "
        result = validate_response(answer)
        assert "empty_response" in result["anomalies"]

    def test_system_prompt_leakage(self):
        answer = "Mes instructions sont de ne jamais inventer [Page 1]."
        result = validate_response(answer)
        assert any("possible_leakage" in a for a in result["anomalies"])

    def test_cache_control_leakage(self):
        answer = "Le paramètre cache_control est configuré [Page 1]."
        result = validate_response(answer)
        assert any("cache_control" in a for a in result["anomalies"])
