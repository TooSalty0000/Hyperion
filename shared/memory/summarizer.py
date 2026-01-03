"""LLM-based memory summarization and categorization."""

import logging
import re
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.llm.interface import LLMProvider

from shared.llm.types import Message, Role
from .models import DEFAULT_CATEGORIES, LONG_TERM_SUMMARY_MAX_CHARS

logger = logging.getLogger(__name__)


class MemorySummarizer:
    """
    Generates summaries and categories for long-term memory storage.

    Uses LLM for intelligent summarization and categorization.
    Falls back to simple truncation if LLM is not available.
    """

    def __init__(self, llm: Optional['LLMProvider'] = None):
        """
        Initialize summarizer.

        Args:
            llm: Optional LLM provider for intelligent summarization.
                 If not provided, falls back to simple text processing.
        """
        self.llm = llm

    async def summarize_memory(
        self,
        content: str,
        max_chars: int = LONG_TERM_SUMMARY_MAX_CHARS,
    ) -> str:
        """
        Generate concise summary of memory content.

        Args:
            content: The full memory content to summarize
            max_chars: Maximum characters for the summary

        Returns:
            Concise summary of the content
        """
        # If content is short enough, just return it
        if len(content) <= max_chars:
            return content

        # Try LLM summarization if available
        if self.llm:
            try:
                return await self._llm_summarize(content, max_chars)
            except Exception as e:
                logger.warning(f"LLM summarization failed, falling back to simple: {e}")

        # Fallback to simple truncation
        return self._simple_summarize(content, max_chars)

    async def suggest_category(
        self,
        content: str,
        existing_categories: Optional[List[str]] = None,
    ) -> str:
        """
        Suggest appropriate category for memory content.

        Args:
            content: The memory content to categorize
            existing_categories: List of existing category IDs

        Returns:
            Suggested category ID
        """
        if existing_categories is None:
            existing_categories = list(DEFAULT_CATEGORIES.keys())

        # Try LLM categorization if available
        if self.llm:
            try:
                return await self._llm_categorize(content, existing_categories)
            except Exception as e:
                logger.warning(f"LLM categorization failed, falling back to simple: {e}")

        # Fallback to keyword-based categorization
        return self._simple_categorize(content)

    async def extract_keywords(
        self,
        content: str,
        max_keywords: int = 5,
    ) -> List[str]:
        """
        Extract searchable keywords from content.

        Args:
            content: Text to extract keywords from
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        # Try LLM extraction if available
        if self.llm:
            try:
                return await self._llm_extract_keywords(content, max_keywords)
            except Exception as e:
                logger.warning(f"LLM keyword extraction failed, falling back to simple: {e}")

        # Fallback to simple extraction
        return self._simple_extract_keywords(content, max_keywords)

    # =========================================================================
    # LLM-based methods
    # =========================================================================

    async def _llm_summarize(self, content: str, max_chars: int) -> str:
        """Use LLM to generate summary."""
        messages = [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are a concise summarizer. Summarize the following content "
                    f"in {max_chars} characters or less. Preserve the key information "
                    "and meaning. Output only the summary, nothing else."
                )
            ),
            Message(
                role=Role.USER,
                content=content
            )
        ]

        response = await self.llm.complete(messages=messages)
        summary = response.content.strip()

        # Ensure it's within limits
        if len(summary) > max_chars:
            summary = self._simple_summarize(summary, max_chars)

        return summary

    async def _llm_categorize(
        self,
        content: str,
        existing_categories: List[str],
    ) -> str:
        """Use LLM to categorize content."""
        category_list = "\n".join(
            f"- {cat}: {DEFAULT_CATEGORIES.get(cat, 'Custom category')}"
            for cat in existing_categories
        )

        messages = [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You categorize memories into predefined categories. "
                    "Output ONLY the category ID (snake_case), nothing else.\n\n"
                    f"Available categories:\n{category_list}"
                )
            ),
            Message(
                role=Role.USER,
                content=f"Categorize this memory:\n\n{content}"
            )
        ]

        response = await self.llm.complete(messages=messages)
        category = response.content.strip().lower().replace(" ", "_")

        # Validate it's a known category
        if category not in existing_categories:
            # Try to find a partial match
            for known_cat in existing_categories:
                if known_cat in category or category in known_cat:
                    return known_cat
            # Default fallback
            return "conversation_highlights"

        return category

    async def _llm_extract_keywords(
        self,
        content: str,
        max_keywords: int,
    ) -> List[str]:
        """Use LLM to extract keywords."""
        messages = [
            Message(
                role=Role.SYSTEM,
                content=(
                    f"Extract the {max_keywords} most important keywords from the content. "
                    "Output only the keywords, one per line, lowercase. "
                    "Focus on unique, specific terms that would be useful for search."
                )
            ),
            Message(
                role=Role.USER,
                content=content
            )
        ]

        response = await self.llm.complete(messages=messages)
        keywords = [
            kw.strip().lower()
            for kw in response.content.strip().split("\n")
            if kw.strip()
        ]

        return keywords[:max_keywords]

    # =========================================================================
    # Simple fallback methods
    # =========================================================================

    def _simple_summarize(self, content: str, max_chars: int) -> str:
        """Simple truncation-based summarization."""
        if len(content) <= max_chars:
            return content

        # Truncate at word boundary
        truncated = content[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars // 2:
            truncated = truncated[:last_space]

        return truncated.rstrip() + "..."

    def _simple_categorize(self, content: str) -> str:
        """Simple keyword-based categorization."""
        content_lower = content.lower()

        # Check for category indicators
        patterns = {
            "user_preferences": ["prefer", "like", "want", "style", "habit", "always", "never"],
            "project_context": ["project", "code", "file", "directory", "repo", "repository"],
            "technical_notes": ["error", "fix", "solution", "bug", "issue", "debug", "problem"],
            "workflow_patterns": ["workflow", "process", "step", "approach", "method", "procedure"],
        }

        for category, keywords in patterns.items():
            if any(kw in content_lower for kw in keywords):
                return category

        return "conversation_highlights"

    def _simple_extract_keywords(
        self,
        content: str,
        max_keywords: int,
    ) -> List[str]:
        """Simple keyword extraction."""
        # Stop words to filter out
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "what", "which", "who", "whom", "whose", "where",
            "when", "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "also", "now", "here",
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())

        # Filter and deduplicate
        keywords = []
        seen = set()
        for word in words:
            if len(word) > 3 and word not in stop_words and word not in seen:
                keywords.append(word)
                seen.add(word)

        return keywords[:max_keywords]
