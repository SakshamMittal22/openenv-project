"""Task definitions for the AI Email Triage Environment."""

from models import EmailCategory, EmailMessage, Priority


def _make_email(
    id: str,
    sender: str,
    subject: str,
    body: str,
    category: EmailCategory,
    priority: Priority,
    reply_keywords: list[str] | None = None,
    timestamp: str = "2025-01-15T09:00:00Z",
) -> EmailMessage:
    return EmailMessage(
        id=id,
        sender=sender,
        subject=subject,
        body=body,
        timestamp=timestamp,
        ground_truth_category=category,
        ground_truth_priority=priority,
        ground_truth_reply_keywords=reply_keywords or [],
    )


TASK_EASY_ID = "easy_spam_detection"

TASK_EASY_EMAILS: list[EmailMessage] = [
    _make_email(
        id="e1",
        sender="promo@cheap-deals.biz",
        subject="YOU WON $1,000,000!!!",
        body="Click here to claim your prize now! No purchase necessary. "
             "Send your bank details immediately.",
        category=EmailCategory.SPAM,
        priority=Priority.LOW,
    ),
    _make_email(
        id="e2",
        sender="hr@company.com",
        subject="Updated PTO Policy",
        body="Hi team, please review the updated paid-time-off policy "
             "attached. Let me know if you have questions.",
        category=EmailCategory.QUERY,
        priority=Priority.MEDIUM,
    ),
    _make_email(
        id="e3",
        sender="offers@unknownstore.xyz",
        subject="Limited time: 99% OFF everything!",
        body="Buy now and save big! Use code SCAM99. Hurry, offer expires "
             "in 1 hour! Unsubscribe link broken.",
        category=EmailCategory.SPAM,
        priority=Priority.LOW,
    ),
    _make_email(
        id="e4",
        sender="manager@company.com",
        subject="Q1 Report Review",
        body="Please send me the Q1 financial summary by end of day Friday.",
        category=EmailCategory.QUERY,
        priority=Priority.MEDIUM,
    ),
]


TASK_MEDIUM_ID = "medium_complaint_handling"

TASK_MEDIUM_EMAILS: list[EmailMessage] = [
    _make_email(
        id="e5",
        sender="angry.customer@gmail.com",
        subject="Terrible service – want a refund",
        body="I ordered product #A123 two weeks ago and it arrived broken. "
             "I called support three times and nobody helped. I demand a full "
             "refund and an apology.",
        category=EmailCategory.COMPLAINT,
        priority=Priority.HIGH,
        reply_keywords=["sorry", "refund", "apolog"],
    ),
    _make_email(
        id="e6",
        sender="curious.user@outlook.com",
        subject="How do I reset my password?",
        body="Hi, I forgot my password and the reset link isn't working. "
             "Can you help?",
        category=EmailCategory.QUERY,
        priority=Priority.MEDIUM,
        reply_keywords=["password", "reset", "help"],
    ),
    _make_email(
        id="e7",
        sender="vip.client@bigcorp.com",
        subject="URGENT: System down in production",
        body="Our production deployment using your API is returning 500 "
             "errors since 08:00 UTC. This is impacting revenue. Need "
             "immediate escalation.",
        category=EmailCategory.URGENT,
        priority=Priority.CRITICAL,
        reply_keywords=["escalat", "investigat", "priorit"],
    ),
]


TASK_HARD_ID = "hard_full_workflow"

TASK_HARD_EMAILS: list[EmailMessage] = [
    _make_email(
        id="e8",
        sender="spam-king@lottery.ru",
        subject="Congratulations! You are selected!",
        body="Dear lucky winner, you have been selected for our exclusive "
             "cash giveaway. Reply with your SSN to claim.",
        category=EmailCategory.SPAM,
        priority=Priority.LOW,
    ),
    _make_email(
        id="e9",
        sender="unhappy.buyer@yahoo.com",
        subject="Wrong item shipped",
        body="I ordered a blue widget (order #W-9981) but received a red "
             "gadget instead. This is the second time this has happened. "
             "Very frustrated.",
        category=EmailCategory.COMPLAINT,
        priority=Priority.HIGH,
        reply_keywords=["sorry", "correct", "ship", "order"],
    ),
    _make_email(
        id="e10",
        sender="cto@partner.io",
        subject="URGENT: Data breach notification",
        body="We detected unauthorized access to shared integration keys "
             "at 03:15 UTC. Immediate action required. Please rotate all "
             "credentials and confirm.",
        category=EmailCategory.URGENT,
        priority=Priority.CRITICAL,
        reply_keywords=["rotat", "credential", "secur", "investigat"],
    ),
    _make_email(
        id="e11",
        sender="new.employee@company.com",
        subject="Onboarding question",
        body="Hi, I'm starting next Monday. Could you let me know what "
             "documents I need to bring on my first day?",
        category=EmailCategory.QUERY,
        priority=Priority.MEDIUM,
        reply_keywords=["document", "welcome", "first day"],
    ),
    _make_email(
        id="e12",
        sender="deals@spammy-marketing.net",
        subject="Act now – miracle weight loss",
        body="Lose 30 lbs in 3 days with our miracle pill! Doctors hate "
             "this trick. Click here. No refunds.",
        category=EmailCategory.SPAM,
        priority=Priority.LOW,
    ),
]


TASK_REGISTRY = {
    TASK_EASY_ID: {
        "emails": TASK_EASY_EMAILS,
        "difficulty": "easy",
        "description": "Classify 4 emails – identify spam vs. legitimate.",
        "required_actions": ["classify"],
    },
    TASK_MEDIUM_ID: {
        "emails": TASK_MEDIUM_EMAILS,
        "difficulty": "medium",
        "description": "Classify 3 emails, reply to complaints/queries/urgent.",
        "required_actions": ["classify", "reply"],
    },
    TASK_HARD_ID: {
        "emails": TASK_HARD_EMAILS,
        "difficulty": "hard",
        "description": "Full workflow: classify, reply, prioritize, resolve 5 emails.",
        "required_actions": ["classify", "reply", "prioritize", "resolve"],
    },
}