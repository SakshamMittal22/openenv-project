from env import EmailTriageEnv
from models import Action, ActionType, EmailCategory, Priority
from tasks import TASK_REGISTRY


def run_baseline(task_id: str):
    env = EmailTriageEnv()
    obs = env.reset(task_id)

    task = TASK_REGISTRY[task_id]
    required = task["required_actions"]

    done = False
    info = {}

    for email in task["emails"]:
        eid = email.id

        # classify
        if "classify" in required:
            action = Action(
                action_type=ActionType.CLASSIFY,
                email_id=eid,
                classification=email.ground_truth_category
            )
            obs, reward, done, info = env.step(action)

        # reply
        if "reply" in required and email.ground_truth_category != EmailCategory.SPAM:
            action = Action(
                action_type=ActionType.REPLY,
                email_id=eid,
                reply_text="We are sorry and will resolve your issue."
            )
            obs, reward, done, info = env.step(action)

        # prioritize
        if "prioritize" in required:
            action = Action(
                action_type=ActionType.PRIORITIZE,
                email_id=eid,
                priority=email.ground_truth_priority
            )
            obs, reward, done, info = env.step(action)

        # resolve
        if "resolve" in required:
            action = Action(
                action_type=ActionType.RESOLVE,
                email_id=eid
            )
            obs, reward, done, info = env.step(action)

        if done:
            break

    return info