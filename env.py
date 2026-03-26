from models import Action, Observation, Reward, EmailStatus, EmailView, EnvState
from tasks import TASK_REGISTRY
from reward import compute_step_reward
from grader import grade_episode


class EmailTriageEnv:
    def __init__(self):
        self.state = EnvState()
        self.required = []
        self.mistakes = []

    def reset(self, task_id=None):
        if task_id is None:
            task_id = list(TASK_REGISTRY.keys())[0]

        task = TASK_REGISTRY[task_id]

        self.state = EnvState(
            task_id=task_id,
            emails=task["emails"],
            statuses=[EmailStatus(email_id=e.id) for e in task["emails"]],
            current_index=0,
            done=False,
            total_reward=0.0,
            step_count=0
        )

        self.required = task["required_actions"]
        self.mistakes = []

        return self._obs("Environment reset")

    def step(self, action: Action):
        email = None
        status = None

        for e, s in zip(self.state.emails, self.state.statuses):
            if e.id == action.email_id:
                email = e
                status = s
                break

        if not email:
            return self._obs("Invalid email"), Reward(value=-0.1), False, {}

        if action.action_type == "classify":
            status.classified_as = action.classification

        elif action.action_type == "reply":
            status.reply_sent = action.reply_text

        elif action.action_type == "prioritize":
            status.assigned_priority = action.priority

        elif action.action_type == "resolve":
            status.resolved = True

        reward = compute_step_reward(action, email, status, self.required)

        if reward.value < 0:
            self.mistakes.append(action.action_type)

        self.state.total_reward += reward.value
        self.state.step_count += 1

        done = self._done()
        self.state.done = done

        info = {"total_reward": self.state.total_reward}

        if done:
            info["grading"] = grade_episode(
                self.state.emails,
                self.state.statuses,
                self.required
            )

        return self._obs("Action applied"), reward, done, info

    def _obs(self, msg):
        current = None
        for e, s in zip(self.state.emails, self.state.statuses):
            if not s.resolved:
                current = EmailView(
                    id=e.id,
                    sender=e.sender,
                    subject=e.subject,
                    body=e.body,
                    timestamp=e.timestamp
                )
                break

        return Observation(
            current_email=current,
            inbox_remaining=sum(1 for s in self.state.statuses if not s.resolved),
            actions_taken=[],
            message=f"{msg} | mistakes: {len(self.mistakes)}"
        )

    def _done(self):
        return all(s.resolved for s in self.state.statuses)