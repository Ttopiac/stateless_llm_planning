# Stateless LLM as Zero-Shot Planners with Trajectory-Aware Prompt and Querying Human Feedback

> **Paper:** [Overleaf Report](https://www.overleaf.com/read/qvqwgjmcrbcn) | **Author:** Chi-Hui Lin

## Key Takeaways

This research investigates how reliable **stateless LLMs** (APIs with no memory between calls) are for **online, zero-shot household task planning**, and proposes three simple, training-free mechanisms to substantially improve their performance.

### Problem: Three Dominant Failure Modes of Stateless LLMs

A naive stateless baseline — prompting the LLM at each step with only the goal, current state, and available actions — consistently fails in three ways:

| Error Mode | Description |
|---|---|
| **Repeated strategies** | The LLM cycles through the same action or short action sequence in a loop |
| **Non-meaningful actions** | The LLM selects actions that leave the environment state unchanged |
| **Over-explanation** | The model interleaves natural language with its action output, producing an invalid response |

These failures cause unnecessary consumption of planning steps and unexpected episode termination.

### Solution: Three Training-Free Mechanisms

1. **Trajectory-aware prompting** — Appends the history of executed actions to each prompt, giving the LLM context about what has already been tried.
2. **LLM-initiated queries** — Instructs the agent to ask a clarifying question instead of guessing when uncertain; the resulting Q&A pairs are replayed in subsequent prompts.
3. **Post-response check with a second chance** — Detects invalid LLM outputs, sends a short corrective feedback message, and lets the model retry before terminating.

### Results

Experiments on **six household tasks** with three frontier LLMs (Gemini-3-Pro, GPT-5.1, Grok-4.1-Fast) in the **TextHouse-Gym** benchmark show:

- **Gemini-3-Pro**: success rate improved from ~10% → **66%**
- **GPT-5.1 and Grok-4.1-Fast**: raised to **near-perfect performance**

### Core Insight

> Three lightweight, prompt-level interventions — trajectory history, on-demand human queries, and output validation — are sufficient to dramatically improve stateless LLM planning without any fine-tuning or internal memory.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ttopiac/llm_world.git
cd llm_world
```
### 2. Create and activate the conda environment

```bash
conda create -n llm_world python=3.12.12 -y
conda activate llm_world
```

This gives you an isolated environment with the exact Python version you expect.

### 3. Install required Python packages

From the repo root (`v` directory):

```bash
pip install "numpy==2.3.5" "pandas==2.3.3" "openai==2.8.1" "gymnasium==1.2.2"
```
