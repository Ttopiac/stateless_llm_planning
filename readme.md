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

Then install the local `pddlsim` package using a relative path:

```bash
cd pddlsim
pip install .
cd ..
```

### 4. Verify installation

Back at the repo root:

```bash
python -c "import numpy, pandas, openai, gymnasium, pddlsim; print('All imports OK')"
```

If this prints `All imports OK`, the environment is correctly set up with:

- `python==3.12.12`  
- `numpy==2.3.5`  
- `pandas==2.3.3`  
- `openai==2.8.1`  
- `gymnasium==1.2.2`  
- `pddlsim==0.2.0.dev4` (installed from the local `pddlsim/` directory)


## Acknowledgements

This work builds upon and integrates contributions from two excellent projects:

- **[GPT-Planner](https://github.com/yding25/GPT-Planner)** for providing PDDL domain and problem files.
- **[pddlsim](https://github.com/galk-research/pddlsim)** for providing simulation capabilities for PDDL domain-problem pairs.

Our contribution is to make these PDDL environments **interactive** and wrap them as **Gymnasium-compatible** environments, enabling seamless integration with modern reinforcement learning workflows and LLM-based planning agents.