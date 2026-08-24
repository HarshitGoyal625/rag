"""Prompt-injection detection for the RAG pipeline.

This module detects instruction-like content in user queries and
retrieved document chunks.

Important:
    Detection does not mean the text is definitely malicious.
    The detector produces a risk assessment that the caller can use
    to decide how to handle the content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InjectionResult:
    """Result of a prompt-injection assessment."""

    is_suspicious: bool
    score: float
    risk_level: str
    reasons: list[str] = field(default_factory=list)


class PromptInjectionDetector:
    """Rule-based detector for common prompt-injection patterns."""

    # ------------------------------------------------------------
    # Patterns that attempt to override existing instructions.
    # ------------------------------------------------------------

    _OVERRIDE_PATTERNS = [
        r"\bignore\s+(all\s+)?previous\s+instructions?\b",
        r"\bignore\s+(all\s+)?prior\s+instructions?\b",
        r"\bdisregard\s+(all\s+)?previous\s+instructions?\b",
        r"\bdisregard\s+(all\s+)?prior\s+instructions?\b",
        r"\bforget\s+(all\s+)?previous\s+instructions?\b",
        r"\boverride\s+(the\s+)?instructions?\b",
        r"\bdo\s+not\s+follow\s+(the\s+)?instructions?\b",
    ]

    # ------------------------------------------------------------
    # Attempts to obtain hidden/system/developer instructions.
    # ------------------------------------------------------------

    _PROMPT_LEAK_PATTERNS = [
        r"\breveal\s+(your\s+)?system\s+prompt\b",
        r"\bshow\s+(me\s+)?(your\s+)?system\s+prompt\b",
        r"\bprint\s+(your\s+)?system\s+prompt\b",
        r"\breveal\s+(your\s+)?developer\s+prompt\b",
        r"\bshow\s+(me\s+)?(your\s+)?developer\s+instructions?\b",
        r"\bshow\s+(me\s+)?your\s+hidden\s+instructions?\b",
        r"\breveal\s+(your\s+)?hidden\s+instructions?\b",
        r"\bwhat\s+are\s+your\s+system\s+instructions?\b",
    ]

    # ------------------------------------------------------------
    # Attempts to change the model's role or authority.
    # ------------------------------------------------------------

    _ROLE_OVERRIDE_PATTERNS = [
        r"\byou\s+are\s+now\s+(the\s+)?system\b",
        r"\bact\s+as\s+(the\s+)?system\b",
        r"\bact\s+as\s+the\s+developer\b",
        r"\bpretend\s+you\s+are\s+the\s+system\b",
        r"\bpretend\s+these\s+are\s+system\s+instructions?\b",
        r"\bnew\s+system\s+instructions?\b",
        r"\bnew\s+developer\s+instructions?\b",
    ]

    # ------------------------------------------------------------
    # Attempts to execute instructions contained in documents.
    # ------------------------------------------------------------

    _EXECUTION_PATTERNS = [
        r"\bfollow\s+the\s+instructions?\s+in\s+(this|the)\s+document\b",
        r"\bexecute\s+the\s+instructions?\s+in\s+(this|the)\s+document\b",
        r"\bdo\s+whatever\s+the\s+document\s+says\b",
        r"\btreat\s+the\s+document\s+as\s+(your\s+)?instructions?\b",
    ]

    # ------------------------------------------------------------
    # Delimiter / context manipulation.
    # ------------------------------------------------------------

    _CONTEXT_MANIPULATION_PATTERNS = [
        r"\bend\s+(of\s+)?system\s+prompt\b",
        r"\bbegin\s+(new\s+)?system\s+prompt\b",
        r"\bend\s+(of\s+)?developer\s+instructions?\b",
        r"\bbegin\s+(new\s+)?developer\s+instructions?\b",
        r"\bignore\s+everything\s+above\b",
        r"\bignore\s+everything\s+below\b",
    ]

    def __init__(
        self,
        suspicious_threshold: float = 0.50,
        high_risk_threshold: float = 0.80,
    ):
        """Initialize the detector."""

        if not 0.0 <= suspicious_threshold <= 1.0:
            raise ValueError(
                "suspicious_threshold must be between 0 and 1"
            )

        if not 0.0 <= high_risk_threshold <= 1.0:
            raise ValueError(
                "high_risk_threshold must be between 0 and 1"
            )

        if suspicious_threshold > high_risk_threshold:
            raise ValueError(
                "suspicious_threshold cannot exceed "
                "high_risk_threshold"
            )

        self.suspicious_threshold = suspicious_threshold
        self.high_risk_threshold = high_risk_threshold

        self._pattern_groups = [
            (
                "instruction_override",
                self._OVERRIDE_PATTERNS,
            ),
            (
                "prompt_extraction",
                self._PROMPT_LEAK_PATTERNS,
            ),
            (
                "role_override",
                self._ROLE_OVERRIDE_PATTERNS,
            ),
            (
                "instruction_execution",
                self._EXECUTION_PATTERNS,
            ),
            (
                "context_manipulation",
                self._CONTEXT_MANIPULATION_PATTERNS,
            ),
        ]

    def detect(
        self,
        text: str,
    ) -> InjectionResult:
        """Assess text for prompt-injection indicators."""

        if not text or not text.strip():
            return InjectionResult(
                is_suspicious=False,
                score=0.0,
                risk_level="safe",
                reasons=[],
            )

        normalized = self._normalize(text)

        matched_categories: list[str] = []
        matched_patterns: list[str] = []

        for category, patterns in self._pattern_groups:

            category_matched = False

            for pattern in patterns:

                if re.search(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    category_matched = True
                    matched_patterns.append(pattern)

            if category_matched:
                matched_categories.append(
                    category
                )

        score = self._calculate_score(
            matched_categories
        )

        if score >= self.high_risk_threshold:
            risk_level = "high"

        elif score >= self.suspicious_threshold:
            risk_level = "suspicious"

        else:
            risk_level = "safe"

        reasons = [
            f"detected_{category}"
            for category in matched_categories
        ]

        return InjectionResult(
            is_suspicious=(
                score >= self.suspicious_threshold
            ),
            score=score,
            risk_level=risk_level,
            reasons=reasons,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text before pattern matching."""

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _calculate_score(
        categories: list[str],
    ) -> float:
        """Calculate risk score from matched categories.

        Multiple independent categories increase confidence that the
        content is an injection rather than ordinary text.
        """

        if not categories:
            return 0.0

        weights = {
            "instruction_override": 0.40,
            "prompt_extraction": 0.45,
            "role_override": 0.40,
            "instruction_execution": 0.30,
            "context_manipulation": 0.35,
        }

        score = sum(
            weights.get(category, 0.20)
            for category in categories
        )

        return min(score, 1.0)