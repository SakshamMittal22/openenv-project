"""
models.py — Typed Pydantic models for the AI Email Triage Environment.

Covers: Actions, Observations, Rewards, State, Tracking (history, mistakes).
Production-grade with full type safety.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class EmailCategory(str, Enum):
    SPAM = "spam"
    COMPLAINT = "complaint"
    QUERY = "query"
    URGENT = "urgent"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    REPLY = "reply"
    PRIORITIZE = "prioritize"
    RESOLVE = "resolve"


# ═══════════════════════════════════════════════════════════════════
# Email
# ═══════════════════════════════════════════════════════════════════

class EmailMessage(BaseModel):
    """Internal email with ground-truth labels (hidden from agent)."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    ground_truth_category: Optional[EmailCategory] = None
    ground_truth_priority: Optional[Priority] = None
    ground_truth_reply_keywords: List[str] = Field(default_factory=list)
    is_ambiguous: bool = Field(
        default=False,
        description="Flag for edge-case / ambiguous emails.",
    )
    difficulty_note: str = Field(
        default="",
        description="Internal note on what makes this email tricky.",
    )


# ═══════════════════════════════════════════════════════════════════
# Agent Action
# ═══════════════════════════════════════════════════════════════════

class Action(BaseModel):
    """A single action submitted by the agent."""
    action_type: ActionType
    email_id: str
    classification: Optional[EmailCategory] = None
    reply_text: Optional[str] = None
    priority: Optional[Priority] = None


# ═══════════════════════════════════════════════════════════════════
# Tracking Records
# ═══════════════════════════════════════════════════════════════════

class ActionRecord(BaseModel):
    """Immutable record of one action taken during the episode."""
    step: int
    action_type: str
    email_id: str
    reward: float
    explanation: str
    correct: Optional[bool] = None


class MistakeRecord(BaseModel):
    """Record of a specific mistake the agent made."""
    step: int
    email_id: str
    action_type: str
    expected: str
    got: str
    penalty: float


# ═══════════════════════════════════════════════════════════════════
# Observation (agent-visible)
# ═══════════════════════════════════════════════════════════════════

class EmailView(BaseModel):
    """Sanitised view of an email — no ground-truth fields."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str


class ProgressInfo(BaseModel):
    """Real-time progress feedback embedded in every observation."""
    total_emails: int = 0
    processed_emails: int = 0
    completion_pct: float = 0.0
    steps_taken: int = 0
    total_reward: float = 0.0
    mistakes_so_far: int = 0
    efficiency_hint: str = ""


class Observation(BaseModel):
    """Returned to the agent after every step() and reset()."""
    current_email: Optional[EmailView] = None
    inbox_remaining: int = 0
    actions_taken: List[str] = Field(default_factory=list)
    message: str = ""
    progress: ProgressInfo = Field(default_factory=ProgressInfo)


# ═══════════════════════════════════════════════════════════════════
# Reward
# ═══════════════════════════════════════════════════════════════════

class Reward(BaseModel):
    """Step-level reward with explainable breakdown."""
    value: float = Field(0.0, description="Net reward, clamped to [-1, 1].")
    breakdown: Dict[str, float] = Field(default_factory=dict)
    explanations: List[str] = Field(
        default_factory=list,
        description="Human-readable reasons for each component.",
    )


# ═══════════════════════════════════════════════════════════════════
# Per-email Processing Status
# ═══════════════════════════════════════════════════════════════════

class EmailStatus(BaseModel):
    """Tracks what the agent has done to a single email."""
    email_id: str
    classified_as: Optional[EmailCategory] = None
    reply_sent: Optional[str] = None
    assigned_priority: Optional[Priority] = None
    resolved: bool = False
    action_count: int = 0
    classify_attempts: int = 0


# ═══════════════════════════════════════════════════════════════════
# Full Environment State
# ═══════════════════════════════════════════════════════════════════

class EnvState(BaseModel):
    """Complete internal state — serialisable for checkpointing."""
    task_id: str = ""
    difficulty: str = ""
    emails: List[EmailMessage] = Field(default_factory=list)
    statuses: List[EmailStatus] = Field(default_factory=list)
    current_index: int = 0
    done: bool = False
    total_reward: float = 0.0
    step_count: int = 0
    max_steps: int = 100
    action_history: List[ActionRecord] = Field(default_factory=list)
    mistakes: List[MistakeRecord] = Field(default_factory=list)