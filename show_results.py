import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.path_utils import PathManager
from envs.pddlgym_envs.pddlsim_env import ErrorType
from typing import Optional, List


def task_model_final_reward(experiment_file: str = 'experiments.csv'):
    experiment_path = PathManager.results_path(experiment_file)

    with open(experiment_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_taken_actions =len(row['taken_actions'].split(' | '))
            print(row['task_id'], row['model_name'], row['final_reward'], num_taken_actions, row['error'])

def show_error_statistics(
    experiment_file: str = 'experiments.csv', 
    output_file: str = 'error_statistics.png',
    selected_models: Optional[List[str]] = None
):
    # Read the CSV file and handle None/empty values properly
    df = pd.read_csv(PathManager.results_path(experiment_file), na_values=[''], keep_default_na=True)

    # Replace NaN with None for consistent handling
    df['error'] = df['error'].fillna('None')

    # Filter for selected models if provided
    if selected_models is not None:
        df = df[df['model_name'].isin(selected_models)]
        if df.empty:
            print(f"No data found for selected models in {experiment_file}")
            return

    # Count occurrences of each error type for each model
    error_counts = df.groupby(['model_name', 'error']).size().unstack(fill_value=0)

    # Ensure all error types are present (even if count is 0)
    error_types_enum = [e for e in ErrorType]
    for error_type in error_types_enum:
        if error_type not in error_counts.columns:
            error_counts[error_type] = 0

    # Reorder columns to match error types
    error_counts = error_counts[error_types_enum]

    error_labels = ["Succeed", "Invalid_Action", "Nonmeaningful_Action", "Exceed_Max_Steps"]
    error_counts.columns = error_labels
    error_types = error_labels

    # If selected_models is provided, reindex the rows to match that exact order
    if selected_models is not None:
        # Filter to only models that actually exist in the data to avoid errors
        existing_models = [m for m in selected_models if m in error_counts.index]
        error_counts = error_counts.reindex(existing_models)

    # Print summary statistics
    print(f"Experiment: {experiment_file}")
    print("Error Statistics Summary:")
    print(error_counts)
    print("\n")

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up bar positions
    models = error_counts.index.tolist()
    x = np.arange(len(models))
    width = 0.2
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    # Plot bars for each error type
    for i, (error_type, label) in enumerate(zip(error_types, error_labels)):
        offset = width * (i - 1.5)
        counts = error_counts[error_type].tolist()
        ax.bar(x + offset, counts, width, label=label, color=colors[i])

    # Customize the plot
    ax.set_xlabel('Model Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Experiments', fontsize=12, fontweight='bold')
    ax.set_title('Error Statistics by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Error Type', loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(PathManager.results_path(output_file), dpi=300, bbox_inches='tight')
    plt.close(fig)

def show_success_rate(
    experiment_file: str = 'experiments.csv', 
    output_file: str = 'success_rate.png',
    selected_models: Optional[List[str]] = None
):
    # Read and process data
    df = pd.read_csv(PathManager.results_path(experiment_file), na_values=[''], keep_default_na=True)
    df['error'] = df['error'].fillna('None')

    # Filter for selected models
    if selected_models is not None:
        df = df[df['model_name'].isin(selected_models)]
        if df.empty:
            print(f"No data found for selected models in {experiment_file}")
            return

    error_counts = df.groupby(['model_name', 'error']).size().unstack(fill_value=0)
    
    error_types_enum = [e for e in ErrorType]
    for error_type in error_types_enum:
        if error_type not in error_counts.columns:
            error_counts[error_type] = 0
    error_counts = error_counts[error_types_enum]
    
    error_labels = ["Succeed", "Invalid_Action", "Nonmeaningful_Action", "Exceed_Max_Steps"]
    error_counts.columns = error_labels

    # Calculate Success Rate and multiply by 100 for percentage
    total_counts = error_counts.sum(axis=1)
    success_rates = (error_counts["Succeed"] / total_counts) * 100
    success_rates = success_rates.fillna(0)

    # Reorder based on selected_models
    if selected_models is not None:
        existing_models = [m for m in selected_models if m in success_rates.index]
        success_rates = success_rates.reindex(existing_models)

    print(f"Experiment: {experiment_file}")
    print("Success Rate Summary (%):")
    print(success_rates)
    print("\n")

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    models = success_rates.index.tolist()
    x = np.arange(len(models))
    width = 0.5
    
    # Generate distinct colors
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(models)))

    rects = ax.bar(x, success_rates, width, label='Success Rate', color=colors)

    # Customize the plot
    ax.set_xlabel('Model Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold') 
    ax.set_title('Success Rate by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    ax.set_ylim(0, 110) 
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', 
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(PathManager.results_path(output_file), dpi=300, bbox_inches='tight')
    plt.close(fig)

def show_average_reward_per_task(
    experiments: dict = {}, 
    output_prefix: str = 'avg_reward_task',
    selected_models: Optional[List[str]] = None
):
    all_data = []

    # 1. Read and aggregate data from all files
    for method_label, filename in experiments.items():
        try:
            file_path = PathManager.results_path(filename)
            df = pd.read_csv(file_path, na_values=[''], keep_default_na=True)
            df['error'] = df['error'].fillna('None')
            
            # Filter: Keep ONLY successful cases
            success_df = df[df['error'] == 'None'].copy()
            
            if success_df.empty:
                print(f"Warning: No successful runs found in {filename}")
                continue

            avg_df = success_df.groupby(['task_id', 'model_name'])['final_reward'].mean().reset_index()
            avg_df['method'] = method_label
            all_data.append(avg_df)
            
        except FileNotFoundError:
            print(f"Error: File {filename} not found at {file_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_data:
        print("No data available to plot.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    tasks = full_df['task_id'].unique()
    method_order = list(experiments.keys())

    for task in tasks:
        print(f"Generating plot for Task: {task}")
        task_df = full_df[full_df['task_id'] == task]
        
        if selected_models is not None:
            task_df = task_df[task_df['model_name'].isin(selected_models)]

        if task_df.empty:
            print(f"No data available for Task {task} with selected models.")
            continue
        
        pivot_df = task_df.pivot(index='model_name', columns='method', values='final_reward')
        pivot_df = pivot_df.reindex(columns=method_order)
        
        if selected_models is not None:
             available_models = [m for m in selected_models if m in pivot_df.index]
             pivot_df = pivot_df.reindex(index=available_models)

        fig, ax = plt.subplots(figsize=(14, 7))
        pivot_df.plot(kind='bar', width=0.8, ax=ax, edgecolor='white', linewidth=0.5)
        
        ax.set_title(f'Average Reward by Model & Method (Success Only) - {task}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model Name', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        output_filename = f"{output_prefix}_{task}.png"
        plt.tight_layout()
        plt.savefig(PathManager.results_path(output_filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_filename}")

def show_success_rate_comparison(
    experiments: dict = {}, 
    output_file: str = 'success_rate_comparison.png',
    selected_models: Optional[List[str]] = None
):
    series_list = []
    method_order = list(experiments.keys())

    for method_label, filename in experiments.items():
        try:
            file_path = PathManager.results_path(filename)
            df = pd.read_csv(file_path, na_values=[''], keep_default_na=True)
            df['error'] = df['error'].fillna('None')
            
            total_counts = df.groupby('model_name').size()
            success_counts = df[df['error'] == 'None'].groupby('model_name').size()
            success_counts = success_counts.reindex(total_counts.index, fill_value=0)
            
            success_rate = (success_counts / total_counts) * 100
            success_rate.name = method_label
            series_list.append(success_rate)
            
        except FileNotFoundError:
            print(f"Error: File {filename} not found at {file_path}")

    if not series_list:
        print("No data available for success rate comparison.")
        return

    combined_df = pd.concat(series_list, axis=1)
    combined_df = combined_df.reindex(columns=method_order)

    if selected_models is not None:
        valid_models = [m for m in selected_models if m in combined_df.index]
        combined_df = combined_df.reindex(index=valid_models)
    
    if combined_df.empty:
        print("Combined dataframe is empty after filtering.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    combined_df.plot(kind='bar', width=0.8, ax=ax, edgecolor='white', linewidth=0.5)
    
    ax.set_title('Success Rate Comparison by Model & Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Name', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(PathManager.results_path(output_file), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_file}")

def show_total_queries_for_active_querying(
    experiment_file: str = 'experiments_trajectory_aware_and_active_queries_v1.csv',
    output_file: str = 'num_queries_active_querying.png',
    selected_models: Optional[List[str]] = None
):
    # Resolve path using your PathManager
    experiment_path = PathManager.results_path(experiment_file)

    # Read CSV
    df = pd.read_csv(experiment_path, na_values=[''], keep_default_na=True)

    # Keep only successful experiments: final_reward > 0
    success_df = df[df['final_reward'] > 0].copy()

    # Optional: filter to specific models
    if selected_models is not None:
        success_df = success_df[success_df['model_name'].isin(selected_models)]

    if success_df.empty:
        print(f"No successful runs (reward > 0) found in {experiment_file}")
        return

    # Count taken actions
    def count_taken_actions(s):
        if pd.isna(s) or str(s).strip() == '':
            return 0
        return len(str(s).split(' | '))

    success_df['num_taken_actions'] = success_df['taken_actions'].apply(count_taken_actions)

    # number_of_queries = (100 - positive_reward + 1) - number_of_taken_actions
    success_df['num_queries'] = (
        100 - success_df['final_reward'] + 1
        - success_df['num_taken_actions']
    )

    # Aggregate total number of queries per model
    stats = success_df.groupby('model_name')['num_queries'].sum().reset_index()

    # Respect model order if provided
    if selected_models is not None:
        order = [m for m in selected_models if m in stats['model_name'].values]
        stats = stats.set_index('model_name').reindex(order).reset_index()

    models = stats['model_name'].tolist()
    num_queries = stats['num_queries'].tolist()

    # ---- Plot: one bar per model, different colors ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))

    # Different color per bar
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(models)))  # or any colormap you like

    ax.bar(x, num_queries, color=colors)

    ax.set_xlabel('Model Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Number of Queries', fontsize=12, fontweight='bold')
    ax.set_title('Total Number of Queries per Model\n(Trajectory-Aware & Active-Querying, Reward > 0)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(PathManager.results_path(output_file), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Total number of queries per model (Trajectory-Aware & Active-Querying, reward > 0):")
    print(stats)
    print(f"Saved plot to: {output_file}")






# ==========================================
# EXECUTION BLOCK
# ==========================================

experiments_dict = {
    'Baseline': 'experiments_v1.csv',
    # 'Trajectory-Aware & Experience-Augmented': 'experiments_experience_augmented_v1.csv',
    'Trajectory-Aware': 'experiments_trajectory_aware_v1.csv',
    'Trajectory-Aware & Active-Querying': 'experiments_trajectory_aware_and_active_queries_v1.csv',
    'Trajectory-Aware & Active-Querying & Post-Check': 'experiments_trajectory_aware_and_active_queries_v2.csv',
}

# Define your specific models and order here
my_selected_models = [
    'google/gemini-3-pro-preview', 
    'openai/gpt-5.1', 
    'x-ai/grok-4.1-fast'
]

# 1. Success Rate Comparison (Aggregated)
show_success_rate_comparison(
    experiments=experiments_dict,
    output_file='success_rate_comparison.png',
    selected_models=my_selected_models
)

# 2. Average Reward Comparison (Per Task)
show_average_reward_per_task(
    experiments=experiments_dict, 
    output_prefix='avg_reward_task', 
    selected_models=my_selected_models
)

# 3. Error Statistics (Individual Files)
# Now looping through files and using the same model filter
for label, filename in experiments_dict.items():
    base_name = filename.replace('.csv', '')
    show_error_statistics(
        experiment_file=filename, 
        output_file=f'error_statistics_{base_name}.png',
        selected_models=my_selected_models
    )

show_total_queries_for_active_querying(
    experiment_file='experiments_trajectory_aware_and_active_queries_v1.csv',
    output_file='num_queries_active_querying.png',
    selected_models=my_selected_models,
)

