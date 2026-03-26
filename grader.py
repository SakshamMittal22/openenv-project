def grade_episode(emails, statuses, required):
    total = len(emails)
    correct = 0

    for email, status in zip(emails, statuses):
        ok = True

        if "classify" in required:
            if status.classified_as != email.ground_truth_category:
                ok = False

        if "reply" in required and email.ground_truth_category != "spam":
            if not status.reply_sent:
                ok = False

        if "prioritize" in required:
            if status.assigned_priority != email.ground_truth_priority:
                ok = False

        if "resolve" in required:
            if not status.resolved:
                ok = False

        if ok:
            correct += 1

    return {"score": correct / total if total else 0}