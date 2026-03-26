from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


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


class EmailMessage(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    ground_truth_category: Optional[EmailCategory] = None
    ground_truth_priority: Optional[Priority] = None
    ground_truth_reply_keywords: List[str] = []


class Action(BaseModel):
    action_type: ActionType
    email_id: str
    classification: Optional[EmailCategory] = None
    reply_text: Optional[str] = None
    priority: Optional[Priority] = None


class EmailView(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str


class Observation(BaseModel):
    current_email: Optional[EmailView] = None
    inbox_remaining: int = 0
    actions_taken: List[str] = []
    message: str = ""


class Reward(BaseModel):
    value: float = 0.0
    breakdown: dict = {}


class EmailStatus(BaseModel):
    email_id: str
    classified_as: Optional[EmailCategory] = None
    reply_sent: Optional[str] = None
    assigned_priority: Optional[Priority] = None
    resolved: bool = False


class EnvState(BaseModel):
    task_id: str = ""
    emails: List[EmailMessage] = []
    statuses: List[EmailStatus] = []
    current_index: int = 0
    done: bool = False
    total_reward: float = 0.0
    step_count: int = 0