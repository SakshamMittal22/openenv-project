from __future__ import annotations

from typing import List, Optional

from models import (
    Action,
    ActionType,
    EmailCategory,
    EmailMessage,
    EmailStatus,
    MistakeRecord,
    Priority,
    Reward,
    ActionRecord,
)


CLASSIFY_CORRECT = 0.25
CLASSIFY_WRONG = -0.15
FIRST_TRY_BONUS = 0.05

REPLY_MAX = 0.25
REPLY_EMPTY_PENALTY = -0.10

REPLY_TONE_MAX = 0.10
REPLY_CONTENT_MAX = 0.15

PRIORITY_CORRECT = 0.20
PRIORITY_CLOSE = 0.08
PRIORITY_WRONG = -0.10

RESOLVE_BONUS = 0.15
RESOLVE_PREMATURE = -0.20

REPEAT_MISTAKE_PENALTY = -0.05
LOOP_PENALTY = -0.10

PRIORITY_RANK = {
    Priority.LOW: 0,
    Priority.MEDIUM: 1,
    Priority.HIGH: 2,
    Priority.CRITICAL: 3,
}


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _confidence_factor(conf: Optional[float]) -> float:
    if conf is None:
        return 1.0
    return 0.7 + 0.6 * float(conf)  # 0.7..1.3


def _keyword_overlap(reply_text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    r = reply_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in r)
    return hits / len(keywords)


def _tone_score(reply_text: str) -> float:
    """
    Returns tone score in [0, 1] using light heuristic signals:
    empathy + professionalism + actionability.
    """
    if not reply_text.strip():
        return 0.0

    lower = reply_text.lower()

    empathy_words = ["sorry", "apolog", "understand", "frustrat", "inconvenien"]
    professionalism_words = ["dear", "hi ", "hello", "thank", "regards", "sincerely", "best", "team"]
    action_words = ["will", "process", "send", "investigat", "escalat", "resolve", "rotate", "audit"]

    empathy_hits = sum(1 for w in empathy_words if w in lower)
    prof_hits = sum(1 for w in professionalism_words if w in lower)
    action_hits = sum(1 for w in action_words if w in lower)

    # Soft thresholds
    empathy = min(1.0, empathy_hits / 2.0)          # up to 0.5..1.0
    prof = min(1.0, prof_hits / 2.0)
    action = min(1.0, action_hits / 2.0)

    # Weighted combine: empathy 0.45, action 0.30, professionalism 0.25
    return round(empathy * 0.45 + action * 0.30 + prof * 0.25, 4)


def _count_actions(
    action_history: List[ActionRecord],
    email_id: str,
    action_type: str,
) -> int:
    return sum(1 for r in action_history if r.email_id == email_id and r.action_type == action_type)


def _count_mistakes(
    mistakes: List[MistakeRecord],
    email_id: str,
    action_type: str,
) -> int:
    return sum(1 for m in mistakes if m.email_id == email_id and m.action_type == action_type)


def compute_step_reward(
    action: Action,
    email: EmailMessage,
    status: EmailStatus,
    required_actions: List[str],
    action_history: List[ActionRecord],
    mistakes: List[MistakeRecord],
) -> Reward:
    breakdown: dict[str, float] = {}
    explanations: List[str] = []
    total = 0.0

    conf_factor = _confidence_factor(action.confidence)

    at = action.action_type

    # Loop detection for repeated same action type on the same email
    prior_same_action = _count_actions(action_history, action.email_id, at.value)
    if prior_same_action >= 3:
        total += LOOP_PENALTY
        breakdown["loop_penalty"] = LOOP_PENALTY
        explanations.append(f"Loop detected: {at.value} repeated {prior_same_action + 1} times.")

    # Classification
    if at == ActionType.CLASSIFY:
        if action.classification is None:
            # Treat missing classification as wrong
            base = CLASSIFY_WRONG * conf_factor
            total += base
            breakdown["classification"] = base
            explanations.append("Missing classification.")
        else:
            correct = action.classification == email.ground_truth_category
            if correct:
                base = CLASSIFY_CORRECT * conf_factor
                total += base
                breakdown["classification"] = base
                explanations.append(f"Correct classification: {action.classification.value}.")

                if status.classify_attempts == 0:
                    total += FIRST_TRY_BONUS
                    breakdown["first_try_bonus"] = FIRST_TRY_BONUS
                    explanations.append("First-try classification bonus.")

            else:
                base = CLASSIFY_WRONG * conf_factor
                total += base
                breakdown["classification"] = base
                explanations.append(
                    f"Wrong classification: got {action.classification.value}, expected {email.ground_truth_category.value}."
                )

                # Repeated mistake penalty for classify on this email
                prior = _count_mistakes(mistakes, action.email_id, "classify")
                if prior > 0:
                    rep_pen = REPEAT_MISTAKE_PENALTY * prior
                    total += rep_pen
                    breakdown["repeat_mistake"] = rep_pen
                    explanations.append(f"Repeated classification mistake penalty (x{prior}).")

    # Reply
    elif at == ActionType.REPLY:
        if "reply" not in required_actions:
            breakdown["reply"] = 0.0
            explanations.append("Reply not required for this task.")
        else:
            if not action.reply_text or not action.reply_text.strip():
                total += REPLY_EMPTY_PENALTY
                breakdown["reply_empty"] = REPLY_EMPTY_PENALTY
                explanations.append("Empty reply penalised.")
            else:
                reply_text = action.reply_text.strip()
                content = _keyword_overlap(reply_text, email.ground_truth_reply_keywords)  # 0..1
                tone = _tone_score(reply_text)  # 0..1

                content_val = round(content * REPLY_CONTENT_MAX, 4)
                tone_val = round(tone * REPLY_TONE_MAX, 4)

                total += content_val + tone_val
                breakdown["reply_content"] = content_val
                breakdown["reply_tone"] = tone_val

                explanations.append(
                    f"Reply scored: content {content:.0%}, tone {tone:.0%}."
                )

    # Prioritize
    elif at == ActionType.PRIORITIZE:
        if "prioritize" not in required_actions:
            breakdown["priority"] = 0.0
            explanations.append("Prioritization not required for this task.")
        elif action.priority is None:
            total += PRIORITY_WRONG
            breakdown["priority"] = PRIORITY_WRONG
            explanations.append("No priority provided.")
        else:
            gt = email.ground_truth_priority
            if gt is not None and action.priority == gt:
                base = PRIORITY_CORRECT * conf_factor
                total += base
                breakdown["priority"] = base
                explanations.append(f"Correct priority: {action.priority.value}.")
            elif gt is not None:
                dist = abs(PRIORITY_RANK[gt] - PRIORITY_RANK[action.priority])
                if dist == 1:
                    base = PRIORITY_CLOSE * conf_factor
                    total += base
                    breakdown["priority"] = base
                    explanations.append(
                        f"Priority close: got {action.priority.value}, expected {gt.value}."
                    )
                else:
                    base = PRIORITY_WRONG * conf_factor
                    total += base
                    breakdown["priority"] = base
                    explanations.append(
                        f"Wrong priority: got {action.priority.value}, expected {gt.value}."
                    )

                prior = _count_mistakes(mistakes, action.email_id, "prioritize")
                if prior > 0:
                    rep_pen = REPEAT_MISTAKE_PENALTY * prior
                    total += rep_pen
                    breakdown["repeat_mistake"] = rep_pen
                    explanations.append(f"Repeated priority mistake penalty (x{prior}).")

    # Resolve
    elif at == ActionType.RESOLVE:
        if "resolve" not in required_actions:
            breakdown["resolve"] = 0.0
            explanations.append("Resolve not required for this task.")
        else:
            if status.classified_as is None:
                total += RESOLVE_PREMATURE
                breakdown["resolve"] = RESOLVE_PREMATURE
                explanations.append("Premature resolve before classification.")
            else:
                total += RESOLVE_BONUS
                breakdown["resolve"] = RESOLVE_BONUS
                explanations.append("Resolved after classification.")

    total = _clamp(total)
    total = round(total, 4)

    return Reward(value=total, breakdown=breakdown, explanations=explanations)