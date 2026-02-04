"""
Prompt template management for BEAM toolkit.

Provides an extensible system for managing agent prompts across
different domains and use cases.
"""

from beam.prompts.registry import PromptRegistry
from beam.prompts.template import PromptTemplate, PromptSet

__all__ = [
    "PromptRegistry",
    "PromptTemplate",
    "PromptSet",
]
