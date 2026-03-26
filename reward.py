from models import Action, EmailMessage, EmailStatus, Reward


def compute_step_reward(action: Action, email: EmailMessage, status: EmailStatus, required):
    score = 0.0
    breakdown = {}

    # Classification
    if action.action_type == "classify":
        if action.classification == email.ground_truth_category:
            score += 0.3
            breakdown["classification"] = 0.3
        else:
            score -= 0.1
            breakdown["classification"] = -0.1

    # Reply
    if action.action_type == "reply":
        if action.reply_text:
            score += 0.3
            breakdown["reply"] = 0.3
        else:
            score -= 0.1
            breakdown["reply"] = -0.1

    # Priority
    if action.action_type == "prioritize":
        if action.priority == email.ground_truth_priority:
            score += 0.2
            breakdown["priority"] = 0.2
        else:
            score -= 0.1
            breakdown["priority"] = -0.1

    # Resolve
    if action.action_type == "resolve":
        score += 0.2
        breakdown["resolve"] = 0.2

    return Reward(value=score, breakdown=breakdown)