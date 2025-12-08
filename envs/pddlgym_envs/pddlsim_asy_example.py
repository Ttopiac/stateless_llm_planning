from pddlsim import parser, simulation

domain, problem = parser.parse_domain_problem_pair_from_files("domain_basic.pddl", "problem_basic.pddl")
sim = simulation.Simulation.from_domain_and_problem(domain, problem)
[print(f"Goal {idx}: {goal}") for idx, goal in enumerate(sim.problem.goals_section)]
reward = 0
    
while not sim.is_solved():
    print(f"Current state: {sim.state._true_predicates}")
    # Get all valid (grounded) actions in the current state
    actions = list(sim.get_grounded_actions())
    if not actions:
        print("No more valid actions! Planning failed or reached dead-end.")
        break

    # For demonstration: pick the first available action (replace with your policy)
    [print(f"{i}: {action}") for i, action in enumerate(actions)]

    action = actions[int(input(f"Which action you would like to take? (0-{len(actions)-1}):"))]

    # Apply the action
    success = sim.apply_grounded_action(action)

    # Print status (optional)
    print(f"Applied: {action}\nstate: {sim.state._true_predicates}\nSuccess: {success}\n")
    reward -= 1
if sim.is_solved():
    reward += 100
print(f"Reward: {reward}")
print("Solved!" if sim.is_solved() else "Not solved.")


