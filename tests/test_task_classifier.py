"""Tests for task classification and adaptive protocol selection.

Matches the style of existing tests in tests/test_engine.py:
- pytest classes grouping related tests
- @pytest.mark.asyncio for async tests
- Descriptive docstrings on every test method
"""
from __future__ import annotations

import pytest

from llm_council.engine.task_classifier import (
    GovernanceProtocol,
    TaskClass,
    TASK_MODEL_PACK,
    TASK_PROTOCOL_MAP,
    classify_task,
    get_protocol_for_task,
)


class TestClassifyTask:
    """Unit tests for the classify_task() function."""

    # ------------------------------------------------------------------ #
    # SECURITY
    # ------------------------------------------------------------------ #

    def test_security_keyword_vulnerability(self):
        """Security keyword 'vulnerability' triggers SECURITY class."""
        result = classify_task("Find all SQL injection vulnerabilities in this codebase")
        assert result == TaskClass.SECURITY

    def test_security_keyword_xss(self):
        """XSS keyword triggers SECURITY class."""
        result = classify_task("Check for XSS attack vectors in the login form")
        assert result == TaskClass.SECURITY

    def test_security_keyword_pentest(self):
        """Pentest keyword triggers SECURITY class."""
        result = classify_task("Perform a pentest on the auth endpoints")
        assert result == TaskClass.SECURITY

    def test_security_keyword_audit(self):
        """Audit keyword triggers SECURITY class."""
        result = classify_task("Security audit the payment processing module")
        assert result == TaskClass.SECURITY

    # ------------------------------------------------------------------ #
    # STRATEGY
    # ------------------------------------------------------------------ #

    def test_strategy_keyword_architect(self):
        """Architect keyword triggers STRATEGY class."""
        result = classify_task("Architect a scalable event-driven microservices system")
        assert result == TaskClass.STRATEGY

    def test_strategy_keyword_adr(self):
        """ADR keyword triggers STRATEGY class."""
        result = classify_task("Write an ADR for switching from REST to GraphQL")
        assert result == TaskClass.STRATEGY

    def test_strategy_keyword_tradeoff(self):
        """Tradeoff keyword triggers STRATEGY class."""
        result = classify_task("Analyse the tradeoff between Redis and Memcached")
        assert result == TaskClass.STRATEGY

    def test_strategy_keyword_should_we(self):
        """'Should we' phrase triggers STRATEGY class."""
        result = classify_task("Should we migrate to a monorepo structure?")
        assert result == TaskClass.STRATEGY

    # ------------------------------------------------------------------ #
    # CODE
    # ------------------------------------------------------------------ #

    def test_code_keyword_implement(self):
        """Implement keyword triggers CODE class."""
        result = classify_task("Implement a JWT refresh token mechanism")
        assert result == TaskClass.CODE

    def test_code_keyword_refactor(self):
        """Refactor keyword triggers CODE class."""
        result = classify_task("Refactor the database access layer to use the repository pattern")
        assert result == TaskClass.CODE

    def test_code_keyword_api(self):
        """API keyword triggers CODE class."""
        result = classify_task("Build a REST API for user profile management")
        assert result == TaskClass.CODE

    def test_code_keyword_write(self):
        """Write keyword triggers CODE class."""
        result = classify_task("Write a function to parse ISO-8601 timestamps")
        assert result == TaskClass.CODE

    # ------------------------------------------------------------------ #
    # REASONING
    # ------------------------------------------------------------------ #

    def test_reasoning_keyword_algorithm(self):
        """Algorithm keyword triggers REASONING class."""
        result = classify_task("Explain the algorithm behind Dijkstra shortest path")
        assert result == TaskClass.REASONING

    def test_reasoning_keyword_complexity(self):
        """Complexity keyword triggers REASONING class."""
        result = classify_task("What is the time complexity of merge sort?")
        assert result == TaskClass.REASONING

    def test_reasoning_keyword_math(self):
        """Math keyword triggers REASONING class."""
        result = classify_task("Solve this math problem: 2^10 + 3^5")
        assert result == TaskClass.REASONING

    def test_reasoning_keyword_calculate(self):
        """Calculate keyword triggers REASONING class."""
        result = classify_task("Calculate the nth Fibonacci number using memoization")
        assert result == TaskClass.REASONING

    # ------------------------------------------------------------------ #
    # RESEARCH
    # ------------------------------------------------------------------ #

    def test_research_keyword_research(self):
        """Research keyword triggers RESEARCH class."""
        result = classify_task("Research the best vector databases for RAG pipelines")
        assert result == TaskClass.RESEARCH

    def test_research_keyword_summarize(self):
        """Summarize keyword triggers RESEARCH class."""
        result = classify_task("Summarize the key differences between gRPC and REST")
        assert result == TaskClass.RESEARCH

    def test_research_keyword_compare(self):
        """Compare keyword triggers RESEARCH class."""
        result = classify_task("Compare Kafka vs RabbitMQ for high-throughput messaging")
        assert result == TaskClass.RESEARCH

    def test_research_keyword_benchmark(self):
        """Benchmark keyword triggers RESEARCH class."""
        result = classify_task("Benchmark PostgreSQL vs DynamoDB for this workload")
        assert result == TaskClass.RESEARCH

    # ------------------------------------------------------------------ #
    # GENERAL (catch-all)
    # ------------------------------------------------------------------ #

    def test_general_no_keywords(self):
        """Task with no matching keywords falls back to GENERAL."""
        result = classify_task("Do the thing")
        assert result == TaskClass.GENERAL

    def test_general_empty_string(self):
        """Empty string falls back to GENERAL."""
        result = classify_task("")
        assert result == TaskClass.GENERAL

    # ------------------------------------------------------------------ #
    # Precedence
    # ------------------------------------------------------------------ #

    def test_security_beats_code(self):
        """SECURITY takes precedence over CODE when both keywords present."""
        result = classify_task("Implement a security audit for the auth module")
        assert result == TaskClass.SECURITY

    def test_security_beats_strategy(self):
        """SECURITY takes precedence over STRATEGY when both keywords present."""
        result = classify_task("Design a security strategy to mitigate XSS attacks")
        assert result == TaskClass.SECURITY

    def test_strategy_beats_code(self):
        """STRATEGY takes precedence over CODE when both keywords present."""
        result = classify_task("Design a system architecture and implement the core module")
        assert result == TaskClass.STRATEGY

    def test_code_beats_reasoning(self):
        """CODE takes precedence over REASONING when both keywords present."""
        result = classify_task("Implement an algorithm to compute the factorial")
        assert result == TaskClass.CODE

    def test_case_insensitive(self):
        """Classification is case-insensitive."""
        assert classify_task("SECURITY audit") == TaskClass.SECURITY
        assert classify_task("IMPLEMENT a function") == TaskClass.CODE
        assert classify_task("CALCULATE the result") == TaskClass.REASONING


class TestGetProtocolForTask:
    """Unit tests for get_protocol_for_task() return values."""

    def test_reasoning_maps_to_majority_vote(self):
        """Reasoning tasks use MAJORITY_VOTE protocol."""
        task_class, protocol = get_protocol_for_task("Calculate the big-O of binary search")
        assert task_class == TaskClass.REASONING
        assert protocol == GovernanceProtocol.MAJORITY_VOTE

    def test_code_maps_to_peer_review_chairman(self):
        """Code tasks use PEER_REVIEW_CHAIRMAN protocol."""
        task_class, protocol = get_protocol_for_task("Implement a Redis caching layer")
        assert task_class == TaskClass.CODE
        assert protocol == GovernanceProtocol.PEER_REVIEW_CHAIRMAN

    def test_security_maps_to_peer_review_chairman(self):
        """Security tasks use PEER_REVIEW_CHAIRMAN protocol."""
        task_class, protocol = get_protocol_for_task("Audit for token injection vulnerabilities")
        assert task_class == TaskClass.SECURITY
        assert protocol == GovernanceProtocol.PEER_REVIEW_CHAIRMAN

    def test_research_maps_to_vote_and_deliberate(self):
        """Research tasks use VOTE_AND_DELIBERATE protocol."""
        task_class, protocol = get_protocol_for_task("Research and summarize the OAuth 2.0 landscape")
        assert task_class == TaskClass.RESEARCH
        assert protocol == GovernanceProtocol.VOTE_AND_DELIBERATE

    def test_strategy_maps_to_hierarchical(self):
        """Strategy tasks use HIERARCHICAL protocol."""
        task_class, protocol = get_protocol_for_task("Architect the data pipeline and decide on storage")
        assert task_class == TaskClass.STRATEGY
        assert protocol == GovernanceProtocol.HIERARCHICAL

    def test_general_maps_to_peer_review_chairman(self):
        """General tasks fall back to PEER_REVIEW_CHAIRMAN."""
        task_class, protocol = get_protocol_for_task("Do the thing with the stuff")
        assert task_class == TaskClass.GENERAL
        assert protocol == GovernanceProtocol.PEER_REVIEW_CHAIRMAN


class TestTaskProtocolMap:
    """Structural tests verifying the TASK_PROTOCOL_MAP is complete."""

    def test_all_task_classes_have_protocol(self):
        """Every TaskClass has an entry in TASK_PROTOCOL_MAP."""
        for task_class in TaskClass:
            assert task_class in TASK_PROTOCOL_MAP, (
                f"{task_class} missing from TASK_PROTOCOL_MAP"
            )

    def test_all_task_classes_have_model_pack(self):
        """Every TaskClass has an entry in TASK_MODEL_PACK."""
        for task_class in TaskClass:
            assert task_class in TASK_MODEL_PACK, (
                f"{task_class} missing from TASK_MODEL_PACK"
            )

    def test_model_pack_has_required_phases(self):
        """Every model pack entry has draft, critique, and synthesis keys."""
        required_phases = {"draft", "critique", "synthesis"}
        for task_class, pack in TASK_MODEL_PACK.items():
            missing = required_phases - pack.keys()
            assert not missing, (
                f"{task_class} model pack missing phases: {missing}"
            )

    def test_protocol_values_are_valid(self):
        """All protocol values in TASK_PROTOCOL_MAP are valid GovernanceProtocol members."""
        valid_protocols = set(GovernanceProtocol)
        for task_class, protocol in TASK_PROTOCOL_MAP.items():
            assert protocol in valid_protocols, (
                f"{task_class} maps to unknown protocol: {protocol}"
            )
