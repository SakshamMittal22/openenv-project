import random
from env import EmailTriageEnv
from models import Action, ActionType, EmailCategory, EmailView, Priority
from tasks import TASK_REGISTRY

_SPAM_KW = [
    "click here", "won", "prize", "lottery", "bank details",
    "miracle", "act now", "99%", "scam", "ssn", "unsubscribe",
    "no refunds", "weight loss", "trick", "guaranteed",
    "c0mpany", "verify your", "suspend",
]
_URGENT_KW = [
    "urgent", "immediately", "critical", "breach", "down",
    "production", "500 error", "escalat", "asap", "data loss",
    "compromis", "unauthori", "incident",
]
_COMPLAINT_KW = [
    "refund", "broken", "frustrated", "angry", "terrible",
    "wrong item", "demand", "disappointed", "complaint",
    "discrepancy", "billing", "shipped wrong",
]


def _classify_with_confidence(view: EmailView):
    text = (view.subject + " " + view.body).lower()
    spam = sum(1 for k in _SPAM_KW if k in text)
    urgent = sum(1 for k in _URGENT_KW if k in text)
    complaint = sum(1 for k in _COMPLAINT_KW if k in text)

    if spam >= 3:
        return EmailCategory.SPAM, 0.95
    if spam >= 2:
        return EmailCategory.SPAM, 0.80
    if urgent >= 3:
        return EmailCategory.URGENT, 0.95
    if urgent >= 2:
        return EmailCategory.URGENT, 0.85
    if complaint >= 2:
        return EmailCategory.COMPLAINT, 0.85
    if complaint >= 1:
        return EmailCategory.COMPLAINT, 0.70
    return EmailCategory.QUERY, 0.65


def _priority(cat: EmailCategory) -> Priority:
    return {
        EmailCategory.SPAM: Priority.LOW,
        EmailCategory.QUERY: Priority.MEDIUM,
        EmailCategory.COMPLAINT: Priority.HIGH,
        EmailCategory.URGENT: Priority.CRITICAL,
    }[cat]


def _reply(view: EmailView, cat: EmailCategory) -> str:
    if cat == EmailCategory.SPAM:
        return ""

    name = view.sender.split("@")[0].replace(".", " ").title()

    if cat == EmailCategory.COMPLAINT:
        return (
            f"Dear {name},\n\n"
            "We are truly sorry for the inconvenience and sincerely apologize for "
            "this frustrating experience. We understand how important this is to you. "
            "We will review your case immediately and process a refund or corrective "
            "shipment as appropriate. Your order will be resolved within 24 hours.\n\n"
            "Best regards,\nCustomer Support Team"
        )

    if cat == EmailCategory.URGENT:
        return (
            f"Dear {name},\n\n"
            "Thank you for alerting us. We understand the urgency and are investigating "
            "this issue with the highest priority. Our security and engineering team "
            "will rotate all credentials immediately, audit access logs, and escalate "
            "as needed. We will send an update within the hour. We are sorry for any "
            "disruption and will work to restore service.\n\n"
            "Best regards,\nIncident Response Team"
        )

    return (
        f"Dear {name},\n\n"
        "Thank you for reaching out. We understand your concern and are happy to help "
        "with your request. For password reset issues, please use the link we will send "
        "shortly or visit our help center. For onboarding documents, please check the "
        "welcome orientation pack. We will review your billing and send a corrective "
        "invoice if needed.\n\n"
        "Best regards,\nSupport Team"
    )


def run_baseline(task_id: str):
    env = EmailTriageEnv()
    env.reset(task_id)
    cfg = TASK_REGISTRY[task_id]
    required = cfg["required_actions"]

    step_log = []
    done = False
    info = {}

    for email_data in cfg["emails"]:
        eid = email_data.id
        view = EmailView(
            id=eid,
            sender=email_data.sender,
            subject=email_data.subject,
            body=email_data.body,
            timestamp=email_data.timestamp,
        )

        cat, conf = _classify_with_confidence(view)

        if "classify" in required:
            obs, reward, done, info = env.step(
                Action(
                    action_type=ActionType.CLASSIFY,
                    email_id=eid,
                    classification=cat,
                    confidence=conf,
                )
            )
            step_log.append(
                {"step": "classify", "email": eid, "reward": reward.value, "confidence": conf}
            )

        if "reply" in required and cat != EmailCategory.SPAM:
            obs, reward, done, info = env.step(
                Action(
                    action_type=ActionType.REPLY,
                    email_id=eid,
                    reply_text=_reply(view, cat),
                )
            )
            step_log.append({"step": "reply", "email": eid, "reward": reward.value})

        if "prioritize" in required:
            obs, reward, done, info = env.step(
                Action(
                    action_type=ActionType.PRIORITIZE,
                    email_id=eid,
                    priority=_priority(cat),
                    confidence=conf,
                )
            )
            step_log.append({"step": "prioritize", "email": eid, "reward": reward.value})

        if "resolve" in required:
            obs, reward, done, info = env.step(
                Action(action_type=ActionType.RESOLVE, email_id=eid)
            )
            step_log.append({"step": "resolve", "email": eid, "reward": reward.value})

        if done:
            break

    if not done:
        from grader import grade_episode
        st = env.get_state()
        info["grading"] = grade_episode(
            st.emails, st.statuses, required, st.action_history, st.mistakes
        )

    grading = info.get("grading", {"score": 0.0})
    return {
        "task_id": task_id,
        "difficulty": cfg["difficulty"],
        "score": grading.get("score", 0.0),
        "breakdown": grading.get("breakdown", {}),
        "details": grading.get("details", []),
        "summary": grading.get("summary", ""),
        "steps": step_log,
        "total_reward": info.get("total_reward", 0.0),
        "mistakes": info.get("mistakes", 0),
    }


def run_random_baseline(task_id: str, seed: int = 42):
    random.seed(seed)
    env = EmailTriageEnv()
    env.reset(task_id)

    cfg = TASK_REGISTRY[task_id]
    required = cfg["required_actions"]

    categories = list(EmailCategory)
    priorities = list(Priority)
    bad_replies = ["ok", "noted", "will do", "thanks", ""]

    step_log = []
    done = False
    info = {}

    for email_data in cfg["emails"]:
        eid = email_data.id
        conf = round(random.uniform(0.3, 1.0), 2)

        if "classify" in required:
            _, reward, done, info = env.step(
                Action(
                    action_type=ActionType.CLASSIFY,
                    email_id=eid,
                    classification=random.choice(categories),
                    confidence=conf,
                )
            )
            step_log.append({"step": "classify", "email": eid, "reward": reward.value})

        if "reply" in required:
            _, reward, done, info = env.step(
                Action(
                    action_type=ActionType.REPLY,
                    email_id=eid,
                    reply_text=random.choice(bad_replies),
                )
            )
            step_log.append({"step": "reply", "email": eid, "reward": reward.value})

        if "prioritize" in required:
            _, reward, done, info = env.step(
                Action(
                    action_type=ActionType.PRIORITIZE,
                    email_id=eid,
                    priority=random.choice(priorities),
                    confidence=conf,
                )
            )
            step_log.append({"step": "prioritize", "email": eid, "reward": reward.value})

        if "resolve" in required:
            _, reward, done, info = env.step(
                Action(action_type=ActionType.RESOLVE, email_id=eid)
            )
            step_log.append({"step": "resolve", "email": eid, "reward": reward.value})

        if done:
            break

    if not done:
        from grader import grade_episode
        st = env.get_state()  # FIX HERE
        info["grading"] = grade_episode(
            st.emails, st.statuses, required, st.action_history, st.mistakes
        )

    grading = info.get("grading", {"score": 0.0})
    return {
        "task_id": task_id,
        "difficulty": cfg["difficulty"],
        "score": grading.get("score", 0.0),
        "breakdown": grading.get("breakdown", {}),
        "steps": step_log,
        "total_reward": info.get("total_reward", 0.0),
        "mistakes": info.get("mistakes", 0),
        "agent": "random",
    }


if __name__ == "__main__":
    print("=" * 72)
    print("  AI Email Triage — Agent Comparison")
    print("  Heuristic Baseline vs Random Agent")
    print("=" * 72)

    for tid in TASK_REGISTRY:
        h = run_baseline(tid)
        r = run_random_baseline(tid)

        gap = h["score"] - r["score"]
        print(f"\n  Task: {tid} ({h['difficulty']})")
        print(f"  {'Metric':<25} {'Heuristic':>10} {'Random':>10} {'Gap':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Score':<25} {h['score']:>10.4f} {r['score']:>10.4f} {gap:>+10.4f}")
        print(f"  {'Reward':<25} {h['total_reward']:>10} {r['total_reward']:>10}")
        print(f"  {'Mistakes':<25} {h['mistakes']:>10} {r['mistakes']:>10}")

        if h.get("breakdown"):
            print(f"\n  Score Breakdown:")
            for k, hv in h["breakdown"].items():
                rv = r.get("breakdown", {}).get(k, 0)
                print(f"    {k:<25} {hv:>8.2%}  vs  {rv:>8.2%}")
    print(f"\n{'=' * 72}")