import gymnasium as gym
from gymnasium import spaces
from enum import StrEnum
from pddlsim import parser, simulation
from pddlsim.ast import GroundedAction  
from typing import Optional, Union

class PDDLSimEnv(gym.Env):
    def __init__(self, domain_file, problem_file):
        self.domain_file = domain_file
        self.problem_file = problem_file
        self._init_sim()
        # You can design a precise observation_space if needed
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)
        self.initial_state = self.get_obs()


    def _init_sim(self):
        self.domain, self.problem = parser.parse_domain_problem_pair_from_files(self.domain_file, self.problem_file)
        self.sim = simulation.Simulation.from_domain_and_problem(self.domain, self.problem)

    def reset(self, *, seed=None, options=None):
        self._init_sim()
        obs = self.get_obs()
        self.initial_state = obs
        legal_actions = list(self._get_actions())
        info = {
            "error": ErrorType.NONE,
            "goal": self.get_goal(),
            "actions": legal_actions,
        }
        return obs, info
    
    def _str_to_action(self, action_str, legal_actions):
        for a in legal_actions:
            if action_str == str(a):
                return a
        return None

    def step(self, action: Optional[Union[str, GroundedAction]]):
        legal_actions = self._get_actions()
        if isinstance(action, str):
            action = self._str_to_action(action, legal_actions)
        if action not in legal_actions:
            reward = -10
            done = True
            obs = self.get_obs()
            info = {
                "error": ErrorType.ILLEGAL_ACTION,
                "goal": self.get_goal(),
                "actions": legal_actions,
            }
            return obs, reward, done, info
        self.sim.apply_grounded_action(action)
        legal_actions = self._get_actions()
        obs = self.get_obs()
        if self.sim.is_solved():
            reward = 100
            done = True
            info = {
                "error": ErrorType.NONE,
                "goal": self.get_goal(),
                "actions": legal_actions,
            }
            return obs, reward, done, info
        elif not legal_actions:
            reward = -100
            done = True
            info = {
                "error": ErrorType.NO_MORE_ACTIONS,
                "goal": self.get_goal(),
                "actions": legal_actions,
            }
            return obs, reward, done, info

        reward = -1
        done = False
        info = {
            "error": ErrorType.NONE,
            "goal": self.get_goal(),
            "actions": legal_actions,
        }
        return obs, reward, done, info

    def get_obs(self):
        # For RL, this should return a structured representation of the current state
        return tuple(self.sim.state._true_predicates)

    def get_goal(self):
        # Assume there is only one goal
        return self.sim.problem.goals_section[0]
    
    def _get_actions(self) -> list[GroundedAction]:
        return list(self.sim.get_grounded_actions())

    def render(self):
        print("Current state:", self.sim.state._true_predicates)

    def build_prompt(self, taken_actions: list[str] = [], additional_prompt: str = "", enable_llm_feedback_queries: bool = False):
        prompt = ""
        prompt += (
            "# SYSTEM ROLE\n"
            "You are a world-class symbolic planning expert. Given the Known Dynamics, select the action from Available Actions that initiates the optimal sequence to satisfy the Goal conditions.\n"
        )
        
        if additional_prompt:
            prompt += additional_prompt

        prompt += (
            "\n# CURRENT PROBLEM\n"
            f"- Goal: {self.get_goal()}\n"
        )
        if len(taken_actions)>0:
            prompt += (
                f"- Initial State: {self.initial_state}\n"
                f"- Taken Actions: {', '.join(taken_actions)}\n"
            )
        prompt += (
            f"- Current State: {self.get_obs()}\n"
            "- Available Actions:\n"
        )
        legal_actions = self._get_actions()
        for i, act in enumerate(legal_actions):
            prompt += f"{i+1}. {act}\n"

        prompt += (
            "\n# INSTRUCTIONS\n"
            "- Select the optimal action from the 'Available Actions' list. Your response must be only the complete action string, including parentheses, with no additional text.\n"
            f"- Valid Answer: {legal_actions[0]}\n"
            f"- Invalid Answer: my action is {legal_actions[0]} because it is promising.\n"
        )
        if enable_llm_feedback_queries:
            prompt += (
                "- **Clarification**: If you cannot confidently determine the correct action, do NOT guess an answer. "
                "Instead, ask a concise question (1-2 sentences) in plain English (no technical term) and end your response must with '?'."
            )
        # prompt += (
        #     "\nYour Response:"            
        # )
        return prompt

class ErrorType(StrEnum):
    """Error types for experiment tracking."""
    NONE = 'None'
    ILLEGAL_ACTION = 'illegal_action'
    NO_MORE_ACTIONS = 'no_more_actions'
    MORE_THAN_MAX_STEPS = 'more_than_max_steps'

if __name__ == "__main__":
    # Example usage:
    env = PDDLSimEnv("domain_basic.pddl", "problem_basic.pddl")
    obs, info = env.reset()
    total_reward = 0
    while True:
        actions = info["actions"]
        print(env.build_prompt())
        chosen_action = str(input())
        # chosen_action = actions[0]
        print("chosen_action: ", chosen_action)
        # print("demooo_action: ", actions[0])
        obs, reward, done, info = env.step(chosen_action)
        total_reward += reward
        print("total_reward: ", total_reward)
        print()
        # env.render()
        if done:
            break
