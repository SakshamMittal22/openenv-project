from enum import Enum
from typing import Dict, List, Optional
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
    ground_truth_reply_keywords: List[str] = Field(default_factory=list)
    is_ambiguous: bool = False
    difficulty_note: str = ""


class Action(BaseModel):
    action_type: ActionType
    email_id: str
    classification: Optional[EmailCategory] = None
    reply_text: Optional[str] = None
    priority: Optional[Priority] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class ActionRecord(BaseModel):
    step: int
    action_type: str
    email_id: str
    reward: float
    explanation: str
    correct: Optional[bool] = None
    confidence: Optional[float] = None


class MistakeRecord(BaseModel):
    step: int
    email_id: str
    action_type: str
    expected: str
    got: str
    penalty: float


class EmailView(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str


class ProgressInfo(BaseModel):
    total_emails: int = 0
    processed_emails: int = 0
    completion_pct: float = 0.0
    steps_taken: int = 0
    total_reward: float = 0.0
    mistakes_so_far: int = 0
    efficiency_hint: str = ""


class Observation(BaseModel):
    current_email: Optional[EmailView] = None
    inbox_remaining: int = 0
    actions_taken: List[str] = Field(default_factory=list)
    message: str = ""
    progress: ProgressInfo = Field(default_factory=ProgressInfo)


class Reward(BaseModel):
    value: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    explanations: List[str] = Field(default_factory=list)


class EmailStatus(BaseModel):
    email_id: str
    classified_as: Optional[EmailCategory] = None
    reply_sent: Optional[str] = None
    assigned_priority: Optional[Priority] = None
    resolved: bool = False
    action_count: int = 0
    classify_attempts: int = 0
    classification_confidence: Optional[float] = None


class EnvState(BaseModel):
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