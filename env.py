import copy
from models import (
    Action, ActionRecord, ActionType, EmailCategory,
    EmailStatus, EmailView, EnvState, MistakeRecord,
    Observation, ProgressInfo, Reward,
)
from grader import grade_episode
from reward import compute_step_reward
from tasks import TASK_REGISTRY


class EmailTriageEnv:

    def __init__(self):
        self._state = EnvState()
        self._required = []

    def reset(self, task_id=None):
        if task_id is None:
            task_id = list(TASK_REGISTRY.keys())[0]
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_id}")
        task = TASK_REGISTRY[task_id]
        emails = copy.deepcopy(task["emails"])
        self._required = task["required_actions"]
        self._state = EnvState(
            task_id=task_id, difficulty=task["difficulty"],
            emails=emails,
            statuses=[EmailStatus(email_id=e.id) for e in emails],
            current_index=0, done=False, total_reward=0.0,
            step_count=0, max_steps=task.get("max_steps", 100),
            action_history=[], mistakes=[])
        return self._build_obs("Environment reset. Process the inbox.")

    def step(self, action):
        if self._state.done:
            return (self._build_obs("Episode finished."),
                    Reward(value=0.0, explanations=["Episode over."]),
                    True, self._build_info())

        email, status = self._find_email(action.email_id)
        if email is None:
            bad = Reward(value=-0.10, breakdown={"error": -0.10},
                         explanations=[f"Email '{action.email_id}' not found."])
            self._state.step_count += 1
            self._state.total_reward += bad.value
            return (self._build_obs(f"Email '{action.email_id}' not found."),
                    bad, False, self._build_info())

        self._apply_action(action, status)

        reward = compute_step_reward(
            action, email, status, self._required,
            self._state.action_history, self._state.mistakes)

        correct = self._check_correct(action, email)
        self._state.action_history.append(ActionRecord(
            step=self._state.step_count,
            action_type=action.action_type.value,
            email_id=action.email_id, reward=reward.value,
            explanation=" | ".join(reward.explanations),
            correct=correct, confidence=action.confidence))

        if correct is False:
            self._log_mistake(action, email)

        self._state.total_reward += reward.value
        self._state.step_count += 1
        status.action_count += 1

        done = self._check_done()

        if self._state.step_count >= self._state.max_steps and not done:
            done = True
            reward = Reward(
                value=max(-1.0, min(1.0, reward.value - 0.30)),
                breakdown={**reward.breakdown, "max_steps": -0.30},
                explanations=reward.explanations + ["Max steps exceeded."])
            self._state.total_reward -= 0.30

        self._state.done = done
        info = self._build_info()

        if done:
            info["grading"] = grade_episode(
                self._state.emails,
                self._state.statuses,
                self._required,
                self._state.action_history,
                self._state.mistakes)

        msg = f"{action.action_type.value} on {action.email_id}: {reward.value:+.4f}"
        return self._build_obs(msg), reward, done, info

    def get_state(self):
        return self._state.model_copy(deep=True)

    def _find_email(self, email_id):
        for e, s in zip(self._state.emails, self._state.statuses):
            if e.id == email_id:
                return e, s
        return None, None

    def _apply_action(self, action, status):
        at = action.action_type
        if at == ActionType.CLASSIFY:
            status.classify_attempts += (1 if status.classified_as else 0)
            if not status.classified_as:
                status.classify_attempts = 1
            status.classified_as = action.classification
            status.classification_confidence = action.confidence
        elif at == ActionType.REPLY:
            status.reply_sent = action.reply_text
        elif at == ActionType.PRIORITIZE:
            status.assigned_priority = action.priority
        elif at == ActionType.RESOLVE:
            status.resolved = True

    def _check_correct(self, action, email):
        if action.action_type == ActionType.CLASSIFY:
            return action.classification == email.ground_truth_category
        if action.action_type == ActionType.PRIORITIZE:
            return action.priority == email.ground_truth_priority
        return None

    def _log_mistake(self, action, email):
        exp, got = "", ""
        if action.action_type == ActionType.CLASSIFY:
            exp = email.ground_truth_category.value if email.ground_truth_category else ""
            got = action.classification.value if action.classification else ""
        elif action.action_type == ActionType.PRIORITIZE:
            exp = email.ground_truth_priority.value if email.ground_truth_priority else ""
            got = action.priority.value if action.priority else ""
        self._state.mistakes.append(MistakeRecord(
            step=self._state.step_count, email_id=action.email_id,
            action_type=action.action_type.value,
            expected=exp, got=got, penalty=-0.15))

    def _check_done(self):
        for email, status in zip(self._state.emails, self._state.statuses):
            if "classify" in self._required and not status.classified_as:
                return False
            if "reply" in self._required:
                if (email.ground_truth_category != EmailCategory.SPAM
                        and not status.reply_sent):
                    return False
            if "prioritize" in self._required and not status.assigned_priority:
                return False
            if "resolve" in self._required and not status.resolved:
                return False
        return True

    def _build_obs(self, message):
        view = None
        emails = self._state.emails
        statuses = self._state.statuses
        n = len(emails)

        for i in range(n):
            idx = (self._state.current_index + i) % n
            if not statuses[idx].resolved:
                self._state.current_index = idx
                e = emails[idx]
                view = EmailView(id=e.id, sender=e.sender,
                                 subject=e.subject, body=e.body,
                                 timestamp=e.timestamp)
                break

        lines = []
        for s in statuses:
            parts = []
            if s.classified_as:
                parts.append(f"classified={s.classified_as.value}")
            if s.reply_sent:
                parts.append("replied")
            if s.assigned_priority:
                parts.append(f"priority={s.assigned_priority.value}")
            if s.resolved:
                parts.append("resolved")
            if parts:
                lines.append(f"{s.email_id}: {', '.join(parts)}")

        processed = sum(1 for s in statuses if s.classified_as or s.resolved)
        unresolved = sum(1 for s in statuses if not s.resolved)
        pct = round(processed / n * 100, 1) if n else 0
        optimal = self._optimal_steps()

        if self._state.step_count > optimal * 1.5:
            hint = "Too many steps."
        elif self._state.step_count <= optimal:
            hint = "Efficient so far."
        else:
            hint = "Slightly above optimal."

        progress = ProgressInfo(
            total_emails=n, processed_emails=processed,
            completion_pct=pct, steps_taken=self._state.step_count,
            total_reward=round(self._state.total_reward, 4),
            mistakes_so_far=len(self._state.mistakes),
            efficiency_hint=hint)

        return Observation(current_email=view, inbox_remaining=unresolved,
                           actions_taken=lines, message=message,
                           progress=progress)

    def _build_info(self):
        return {"task_id": self._state.task_id,
                "difficulty": self._state.difficulty,
                "step_count": self._state.step_count,
                "total_reward": round(self._state.total_reward, 4),
                "mistakes": len(self._state.mistakes)}

    def _optimal_steps(self):
        return sum(1 for e in self._state.emails
                   for a in self._required
                   if not (a == "reply" and
                           e.ground_truth_category == EmailCategory.SPAM))