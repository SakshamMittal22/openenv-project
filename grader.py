from models import (
    ActionRecord, EmailCategory, EmailMessage,
    EmailStatus, MistakeRecord, Priority,
)

PRIORITY_RANK = {
    Priority.LOW: 0, Priority.MEDIUM: 1,
    Priority.HIGH: 2, Priority.CRITICAL: 3,
}


def _classify_score(status, email):
    if status.classified_as is None:
        return 0.0
    return 1.0 if status.classified_as == email.ground_truth_category else 0.0


def _reply_quality(status, email):
    kws = email.ground_truth_reply_keywords

    if email.ground_truth_category == EmailCategory.SPAM:
        return 1.0 if not status.reply_sent else 0.6

    reply = status.reply_sent or ""
    if not reply.strip():
        return 0.0

    if kws:
        hits = sum(1 for k in kws if k.lower() in reply.lower())
        kw_score = hits / len(kws)
    else:
        kw_score = 1.0

    len_score = 1.0 if len(reply.strip()) >= 50 else 0.4

    tone = 0.0
    lower = reply.lower()
    if any(g in lower for g in ["dear", "hi ", "hello", "thank"]):
        tone += 0.4
    if any(w in lower for w in ["sorry", "apolog", "help", "investigat", "resolv"]):
        tone += 0.3
    if any(c in lower for c in ["regards", "sincerely", "best", "team"]):
        tone += 0.3

    return round(kw_score * 0.50 + len_score * 0.20 + tone * 0.30, 4)


def _satisfaction(email, status, action_history):
    """Simulate customer satisfaction survey (CSAT).
    
    Factors: empathy, actionability, professionalism, accuracy, responsiveness.
    Returns 0-1 score representing how satisfied a real user would be.
    """
    if email.ground_truth_category == EmailCategory.SPAM:
        return 1.0 if status.classified_as == EmailCategory.SPAM else 0.2

    reply = status.reply_sent or ""
    if not reply.strip():
        return 0.1

    lower = reply.lower()

    empathy_words = ["sorry", "understand", "apolog", "frustrat", "inconvenien"]
    empathy = min(1.0, sum(0.3 for w in empathy_words if w in lower))

    action_words = ["will", "process", "send", "investigat", "resolv",
                    "ship", "refund", "escalat", "restor", "review"]
    actionability = min(1.0, sum(0.18 for w in action_words if w in lower))

    prof = 0.0
    if any(g in lower for g in ["dear", "hi ", "hello", "thank"]):
        prof += 0.5
    if any(c in lower for c in ["regards", "sincerely", "best", "team"]):
        prof += 0.5

    accuracy = 1.0 if status.classified_as == email.ground_truth_category else 0.3

    email_actions = sum(1 for a in action_history if a.email_id == email.id)
    responsiveness = 1.0 if email_actions <= 4 else max(0.3, 1.0 - (email_actions - 4) * 0.15)

    return round(
        empathy * 0.20 + actionability * 0.25 + prof * 0.15 +
        accuracy * 0.25 + responsiveness * 0.15, 4)


def _priority_score(status, email):
    if not status.assigned_priority or not email.ground_truth_priority:
        return 0.0
    if status.assigned_priority == email.ground_truth_priority:
        return 1.0
    dist = abs(PRIORITY_RANK[email.ground_truth_priority] -
               PRIORITY_RANK[status.assigned_priority])
    return {1: 0.50, 2: 0.25}.get(dist, 0.0)


def _completeness(status, email, required):
    done, total = 0, 0
    if "classify" in required:
        total += 1
        done += int(status.classified_as is not None)
    if "reply" in required and email.ground_truth_category != EmailCategory.SPAM:
        total += 1
        done += int(bool(status.reply_sent))
    if "prioritize" in required:
        total += 1
        done += int(status.assigned_priority is not None)
    if "resolve" in required:
        total += 1
        done += int(status.resolved)
    return done / total if total > 0 else 1.0


def _calibration_score(action_history):
    rated = [a for a in action_history
             if a.confidence is not None and a.correct is not None]
    if not rated:
        return 0.5
    total_error = sum(abs(a.confidence - (1.0 if a.correct else 0.0))
                      for a in rated)
    return round(max(0.0, 1.0 - total_error / len(rated)), 4)


def grade_episode(emails, statuses, required_actions,
                  action_history, mistakes):
    n = len(emails)
    if n == 0:
        return {"score": 0.0, "breakdown": {}, "details": [], "summary": ""}

    has = {c: c in required_actions
           for c in ["classify", "reply", "prioritize", "resolve"]}
    active = [k for k, v in has.items() if v]

    weight_map = {}
    if has["classify"]:
        weight_map["classification"] = 0.25
    if has["reply"]:
        weight_map["reply_quality"] = 0.18
        weight_map["satisfaction"] = 0.14
    if has["prioritize"]:
        weight_map["priority"] = 0.12
    weight_map["completeness"] = 0.14
    weight_map["efficiency"] = 0.09
    weight_map["calibration"] = 0.08

    wt_sum = sum(weight_map.values())
    weight_map = {k: round(v / wt_sum, 4) for k, v in weight_map.items()}

    details = []
    agg = {k: 0.0 for k in weight_map}

    for email, status in zip(emails, statuses):
        row = {"email_id": email.id}

        if "classification" in weight_map:
            s = _classify_score(status, email)
            row["classification"] = s
            agg["classification"] += s

        if "reply_quality" in weight_map:
            s = _reply_quality(status, email)
            row["reply_quality"] = round(s, 4)
            agg["reply_quality"] += s

        if "satisfaction" in weight_map:
            s = _satisfaction(email, status, action_history)
            row["satisfaction"] = round(s, 4)
            agg["satisfaction"] += s

        if "priority" in weight_map:
            s = _priority_score(status, email)
            row["priority"] = round(s, 4)
            agg["priority"] += s

        comp = _completeness(status, email, required_actions)
        row["completeness"] = round(comp, 4)
        agg["completeness"] += comp
        details.append(row)

    for k in agg:
        if k not in ("efficiency", "calibration"):
            agg[k] = round(agg[k] / n, 4)

    optimal = sum(1 for e in emails for a in active
                  if not (a == "reply" and
                         e.ground_truth_category == EmailCategory.SPAM))
    actual = len(action_history) if action_history else optimal
    agg["efficiency"] = round(min(1.0, optimal / max(actual, 1)), 4)

    agg["calibration"] = _calibration_score(action_history)

    final = sum(agg[k] * weight_map[k] for k in weight_map)
    final = round(max(0.0, min(1.0, final)), 4)

    parts = [f"{k}: {v:.2%}" for k, v in agg.items()]
    summary = f"Score {final:.4f} | " + " | ".join(parts)

    return {
        "score": final,
        "breakdown": {k: agg[k] for k in weight_map},
        "weights": weight_map,
        "details": details,
        "mistakes_total": len(mistakes),
        "steps_taken": len(action_history),
        "optimal_steps": optimal,
        "summary": summary,
    }