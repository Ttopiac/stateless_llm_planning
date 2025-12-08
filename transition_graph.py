import pandas as pd
from collections import defaultdict
from utils.path_utils import PathManager
import re


def normalize_state(state_str):
    """
    Normalize a PDDL state by sorting predicates alphabetically.
    Handles both comma-separated and space-separated predicates.
    
    Example:
    "((robot_at rob kitchen), (vacuum_at vacuum_1 dining))" 
    -> "((robot_at rob kitchen) (vacuum_at vacuum_1 dining))"
    """
    if pd.isna(state_str):
        return state_str
    
    # Extract individual predicates using regex
    # Match patterns like (predicate args)
    predicates = re.findall(r'\([^()]+\)', str(state_str))
    
    # Clean and sort predicates
    predicates = [p.strip() for p in predicates]
    predicates.sort()
    
    # Reconstruct normalized state
    normalized = "(" + " ".join(predicates) + ")"
    return normalized


def get_state_transitions(transitions_df, task_id, current_state):
    """
    Extract transition information for a specific state in a deterministic environment.
    Each action maps to exactly one (next_state, reward) pair.
    """
    # Normalize the query state
    normalized_query_state = normalize_state(current_state)
    
    # Create a normalized state column if it doesn't exist
    if 'normalized_current_state' not in transitions_df.columns:
        transitions_df['normalized_current_state'] = transitions_df['current_state'].apply(normalize_state)
        transitions_df['normalized_next_state'] = transitions_df['next_state'].apply(normalize_state)
    
    # Filter for the specific task and normalized current state
    state_data = transitions_df[
        (transitions_df['task_id'] == task_id) & 
        (transitions_df['normalized_current_state'] == normalized_query_state)
    ]
    
    if state_data.empty:
        return None
    
    # Get goal and legal actions
    goal = state_data.iloc[0]['goal']
    legal_actions = state_data.iloc[0]['legal_actions']
    
    # Get unique action → (next_state, reward) mappings
    transitions = {}
    for _, row in state_data.iterrows():
        action = row['planned_action']
        if action not in transitions:  # Take first occurrence (all should be same)
            transitions[action] = {
                'next_state': row['next_state'],  # Keep original formatting
                'reward': row['reward'],
                'error': row['error'] if pd.notna(row['error']) else None
            }
    
    return {
        'task_id': task_id,
        'goal': goal,
        'current_state': current_state,  # Return original query format
        'legal_actions': legal_actions,
        'transitions': transitions
    }


def format_state_transitions_for_llm(state_info, only_transition_info=False):
    """
    Format state transition info as text for LLM consumption.
    Groups multiple actions under a single shared Current State.
    """
    if not state_info:
        return "No transitions found for this state."
    
    lines = []
    lines.append("\n# KNOWN DYNAMIXS (PAST EXPERIENCE)")
    lines.append("Use the following observed transitions to understand the consequence of your actions:\n")
    
    # 1. State the context once
    lines.append("From the Current State:")
    lines.append(f"{state_info['current_state']}\n")
    
    lines.append("The following outcomes were observed:\n")
    
    # 2. List the variable actions/outcomes
    id = 1
    for action, outcome in state_info['transitions'].items():
        lines.append(f"{id}. Action: {action}")
        lines.append("   Outcome:")
        lines.append(f"     - Resulted State: {outcome['next_state']}")
        lines.append(f"     - Reward: {outcome['reward']}\n")
        id += 1
    
    return "\n".join(lines)



def state_transition_prompt(df, task_id, current_state, only_transition_info=True):
    state_info = get_state_transitions(df, task_id, current_state)
    return format_state_transitions_for_llm(state_info, only_transition_info)


if __name__ == "__main__":
    # Usage Example
    df = pd.read_csv(PathManager.results_path("transitions.csv"))
    
    # Test with different orderings of the same state
    # current_state = "((vacuum_at vacuum_1 dining), (vacuum_is_unplugged vacuum_1), (robot_at rob kitchen), (outlet_at outlet_1 dining), (table_at table_0 dining), (vacuum_is_off vacuum_1))"
    current_state = "((plate_at plate_1 kitchen), (fork_at fork_1 kitchen), (cupboard_is_found cupboard_0), (table_at table_0 dining), (hand_empty rob), (table_is_found table_0), (cupboard_is_closed cupboard_0), (cupboard_at cupboard_0 kitchen), (robot_at rob dining))"
    # This should now work even if order differs in CSV
    llm_prompt = state_transition_prompt(df=df, task_id=1, current_state=current_state, only_transition_info=False)
    print(llm_prompt)
    
    # Verify normalization works
    state1 = "((robot_at rob kitchen) (vacuum_at vacuum_1 dining) (table_at table_0 dining))"
    state2 = "((vacuum_at vacuum_1 dining), (robot_at rob kitchen), (table_at table_0 dining))"
    print(f"\nNormalized state1: {normalize_state(state1)}")
    print(f"Normalized state2: {normalize_state(state2)}")
    print(f"Are they equal? {normalize_state(state1) == normalize_state(state2)}")
