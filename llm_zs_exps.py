from __future__ import annotations

import csv
import pandas as pd
from itertools import product
from pathlib import Path
from dataclasses import dataclass

from utils.path_utils import PathManager
from utils.openrouter_helper import OpenRouterClient
from envs.pddlgym_envs.pddlsim_env import PDDLSimEnv, ErrorType
from transition_graph import state_transition_prompt


# Define fieldnames once as module-level constants
EXPERIMENTS_FIELDNAMES = [
    "task_id",
    "model_name",
    "goal",
    "initial_state",
    "taken_actions",
    "final_state",
    "error",
    "final_reward",
]

TRANSITIONS_FIELDNAMES = [
    "task_id",
    "model_name",
    "goal",
    "initial_state",
    "current_state",
    "legal_actions",
    "planned_action",
    "next_state",
    "error",
    "reward",
    "taken_actions",
    "final_reward",
]


@dataclass
class ExperimentConfig:
    """Configuration for experiment logging paths."""
    experiments_log_path: Path
    transitions_log_path: Path
    transitions_history_path: Path | None = None


def ensure_log_headers(config: ExperimentConfig) -> None:
    """
    Make sure both CSV files exist and have header rows.
    """
    # Ensure experiments.csv
    config.experiments_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.experiments_log_path.exists():
        with config.experiments_log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_FIELDNAMES)
            writer.writeheader()
    
    # Ensure transitions.csv
    config.transitions_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.transitions_log_path.exists():
        with config.transitions_log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRANSITIONS_FIELDNAMES)
            writer.writeheader()


def log_transition(
    config: ExperimentConfig,
    task_id: int,
    model_name: str,
    goal: str,
    initial_state: str,
    current_state: str,
    legal_actions: list,
    planned_action: str,
    next_state: str,
    error: str | None,
    reward: float,
    taken_actions: list[str],
    final_reward: float,
) -> dict:
    """
    Log a single transition to transitions.csv and return the record.
    """
    transition_record = {
        "task_id": task_id,
        "model_name": model_name,
        "goal": goal,
        "initial_state": initial_state,
        "current_state": current_state,
        "legal_actions": " | ".join(str(a) for a in legal_actions),
        "planned_action": planned_action,
        "next_state": next_state,
        "error": error,
        "reward": reward,
        "taken_actions": " | ".join(taken_actions),
        "final_reward": final_reward,
    }
    
    with config.transitions_log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRANSITIONS_FIELDNAMES)
        writer.writerow(transition_record)
    
    return transition_record


def log_experiment(
    config: ExperimentConfig,
    task_id: int,
    model_name: str,
    goal: str,
    initial_state: str,
    taken_actions: list[str],
    final_state: str,
    error: str | None,
    final_reward: float,
) -> dict:
    """
    Log an experiment summary to experiments.csv and return the record.
    """
    experiment_record = {
        "task_id": task_id,
        "model_name": model_name,
        "goal": goal,
        "initial_state": initial_state,
        "taken_actions": " | ".join(taken_actions),
        "final_state": final_state,
        "error": error,
        "final_reward": final_reward,
    }
    
    with config.experiments_log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_FIELDNAMES)
        writer.writerow(experiment_record)
    
    return experiment_record


def run_single_experiment(
    config: ExperimentConfig,
    task_id: int,
    model_name: str,
    client: OpenRouterClient,
    *,
    verbose: bool = True,
    max_steps: int = 20,
    trajectory_aware: bool = False,
    experience_augmented: bool = False,
    enable_llm_feedback_queries: bool = False,
) -> None:
    """
    Run one episode for (task_id, model_name), log the result as one row.
    """
    domain_path = PathManager.task_path(task_id, "domain_basic.pddl")
    problem_path = PathManager.task_path(task_id, "problem_basic.pddl")

    env = PDDLSimEnv(domain_path, problem_path)
    obs, info = env.reset()
    initial_state = obs
    total_reward = 0
    goal = env.get_goal()
    taken_actions: list[str] = []
    steps = 0
    error: str | None = None

    if experience_augmented:
        if config.transitions_history_path is None:
            raise ValueError("transitions_history_path must be set when experience_augmented=True")
        df = pd.read_csv(config.transitions_history_path)
    else:
        df = None

    feedback_prompt = ""
    while True:
        current_state = obs
        legal_actions = info["actions"]
        
        if experience_augmented and df is not None:
            additional_prompt = state_transition_prompt(
                df=df, 
                task_id=int(task_id), 
                current_state=current_state, 
                only_transition_info=True
            )
        else:
            additional_prompt = ""
        if trajectory_aware:
            prompt = env.build_prompt(
                taken_actions=taken_actions, 
                additional_prompt=additional_prompt, 
                enable_llm_feedback_queries=enable_llm_feedback_queries
            )
        else: 
            prompt = env.build_prompt(
                additional_prompt=additional_prompt, 
                enable_llm_feedback_queries=enable_llm_feedback_queries
            )
        
        if feedback_prompt and enable_llm_feedback_queries:
            prompt += feedback_prompt

        if verbose or model_name == "Human":
            print(f"\n[Task {task_id} | Model {model_name}] step {steps}")
            print(prompt + "\nYour Response:\n")
        
        if model_name == "Human":
            llm_output = input()
        else:
            llm_output = client.chat(model_name, prompt + "\nYour Response:")

        if verbose:
            print("LLM output:", llm_output)
        
        if enable_llm_feedback_queries and llm_output.strip().endswith("?"):
            print(f"\n[Task {task_id} | Model {model_name}] step {steps}")
            print(prompt)
            print("Model's Response:\n", llm_output)
            print("USER FEEDBACK:")
            user_feedback = input()
            prompt = env.build_prompt(
                additional_prompt=additional_prompt, 
                enable_llm_feedback_queries=False,
            )
            current_feedback = ""
            current_feedback += (f"\nPrevious Question: {llm_output}\n")
            current_feedback += ("USER FEEDBACK: " + user_feedback + "\n")
            feedback_prompt += (f"{current_feedback}")
            prompt += (f"{feedback_prompt}")
            prompt +=( "\nNow, select a valid action without any explanation:")
            total_reward -= 1
            if model_name == "Human":
                print(prompt)
                llm_output = input()
            else:
                llm_output = client.chat(model_name, prompt)
            print(prompt)
            print("LLM output:", llm_output)
        else:
            if isinstance(llm_output, str):
                llm_action = env._str_to_action(llm_output, legal_actions)
                if llm_action not in legal_actions:
                    prompt = env.build_prompt(
                        additional_prompt=additional_prompt, 
                        enable_llm_feedback_queries=enable_llm_feedback_queries,
                    )
                    current_feedback = ""
                    current_feedback += (f"\nPrevious Answer: {llm_output[:min(len(llm_output), 100)]}\n")
                    current_feedback += ("USER FEEDBACK: Your provided solution is not Valid.(You may explain too much)\n")
                    feedback_prompt += (f"{current_feedback}")
                    prompt += (f"{feedback_prompt}")
                    prompt +=( "\nNow, select a valid action without any explanation:")
                    total_reward -= 10
                    if model_name == "Human":
                        print(prompt)
                        llm_output = input()
                    else:
                        print(current_feedback)
                        llm_output = client.chat(model_name, prompt)
                    print(prompt)
                    print("LLM output:", llm_output)

            
        obs, reward, done, info = env.step(llm_output)
        taken_actions.append(str(llm_output))
        total_reward += reward
        steps += 1

        # Log this transition
        transition_record = log_transition(
            config=config,
            task_id=task_id,
            model_name=model_name,
            goal=goal,
            initial_state=initial_state,
            current_state=current_state,
            legal_actions=legal_actions,
            planned_action=llm_output,
            next_state=obs,
            error=info.get("error") if done else None,
            reward=reward,
            taken_actions=taken_actions,
            final_reward=total_reward,
        )

        if verbose:
            print(
                f"Chosen action: {llm_output}, "
                f"Reward: {reward}, Total reward: {total_reward}\n"
            )
            print("Logged transition record:", transition_record)

        if steps > max_steps:
            error = ErrorType.MORE_THAN_MAX_STEPS
            break

        if done:
            error = info.get("error")
            break

    # Log experiment
    experiment_record = log_experiment(
        config=config,
        task_id=task_id,
        model_name=model_name,
        goal=goal,
        initial_state=initial_state,
        taken_actions=taken_actions,
        final_state=obs,
        error=error,
        final_reward=total_reward,
    )

    if verbose:
        print("Logged experiment record:", experiment_record)


def run_experiments(
    config: ExperimentConfig,
    task_ids: list[int],
    llm_models: list[str],
    client: OpenRouterClient,
    *,
    verbose: bool = False,
    trajectory_aware: bool = False,
    experience_augmented: bool = False,
    enable_llm_feedback_queries: bool = False,
) -> None:
    """
    Run experiments for all combinations of task_ids and llm_models.
    """
    ensure_log_headers(config)
    
    for task_id, model_name in product(task_ids, llm_models):
        if verbose:
            print(f"\n=== Running experiment: task {task_id}, model {model_name} ===")
        run_single_experiment(
            config=config,
            task_id=task_id,
            model_name=model_name,
            client=client,
            verbose=verbose,
            max_steps=PathManager.MAX_STEPS[task_id],
            trajectory_aware=trajectory_aware,
            experience_augmented=experience_augmented,
            enable_llm_feedback_queries=enable_llm_feedback_queries,
        )


def main() -> None:
    """Main entry point with experiment configuration."""
    
    # Choose your experiment type by uncommenting the appropriate config
    test = True
    
    trajectory_aware = True
    experience_augmented = False
    enable_llm_feedback_queries = True
    
    # Config 1: Experiments with action history only
    if test: 
        config = ExperimentConfig(
            experiments_log_path=PathManager.results_path("experiments_test.csv"),
            transitions_log_path=PathManager.results_path("transitions_test.csv"),
            transitions_history_path=PathManager.results_path("transitions.csv")
        )
    elif trajectory_aware:
        if not experience_augmented:
            if not enable_llm_feedback_queries:
                config = ExperimentConfig(
                    experiments_log_path=PathManager.results_path("experiments_trajectory_aware_v1.csv"),
                    transitions_log_path=PathManager.results_path("transitions_trajectory_aware_v1.csv"),
                )
            else:
                config = ExperimentConfig(
                    experiments_log_path=PathManager.results_path("experiments_trajectory_aware_and_active_queries_v2.csv"),
                    transitions_log_path=PathManager.results_path("transitions_trajectory_aware_and_active_queries_v2.csv"),
                )
        else:
            if not enable_llm_feedback_queries:
                config = ExperimentConfig(
                    experiments_log_path=PathManager.results_path("experiments_experience_augmented_v1.csv"),
                    transitions_log_path=PathManager.results_path("transitions_experience_augmented_v1.csv"),
                    transitions_history_path=PathManager.results_path("transitions_v1.csv"),
                )
            else:
                config = None
    elif not experience_augmented and not enable_llm_feedback_queries:
        config = ExperimentConfig(
            experiments_log_path=PathManager.results_path("experiments_v1.csv"),
            transitions_log_path=PathManager.results_path("transitions_v1.csv"),
        )
    else:
        config = None

    client = OpenRouterClient()

    # All task IDs and models to sweep over
    # task_ids = sorted(PathManager.VALID_TASK_IDS)
    task_ids = [10,11]
    llm_models = [
        "Human",
        OpenRouterClient.GPT_5,
        OpenRouterClient.GROK_4_1_FAST,
        OpenRouterClient.GROK_4_1_FAST,
        OpenRouterClient.GROK_4_1_FAST,
        OpenRouterClient.GEMINI_3_PRO,
        OpenRouterClient.GEMINI_3_PRO,
        OpenRouterClient.GPT_5,
        OpenRouterClient.GPT_5,
    ]


    run_experiments(
        config=config,
        task_ids=task_ids,
        llm_models=llm_models,
        client=client,
        verbose=True or test,
        trajectory_aware=trajectory_aware,
        experience_augmented=experience_augmented,
        enable_llm_feedback_queries=enable_llm_feedback_queries,
    )


if __name__ == "__main__":
    main()
