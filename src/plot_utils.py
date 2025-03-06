import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_fairness_metrics(data, label_pos, label_fair_pos_metrics, label_fair_di_pos, label_fair_index=None):
    """
    Plot the fairness metrics and disparate impact against the penalty.
    """
    if label_fair_index is None:
        label_fair_index = data.index.max()

    # Set up the color palette
    palette = sns.color_palette("husl", len(data.columns))
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plotting metrics except disparate impact
    non_disparate_metrics = data.columns[data.columns != "disparate_impact"]
    for i, metric in enumerate(non_disparate_metrics):
        axes[0].plot(data.index, data[metric], label=metric.replace('_', ' ').title(), color=palette[i])

    # Add a horizontal dotted line at y=0 with label "Fair"
    axes[0].axhline(y=0, color='black', linestyle='--')
    axes[0].text(x=label_fair_index, y=label_fair_pos_metrics, s="Fair", color='black', ha='left', va='center', fontsize=10)

    # Set y-axis limit between -1 and 1 for the first plot
    axes[0].set_ylim(-1.05, 1.05)

    axes[0].set_xlabel("Penalty", fontsize=12)
    axes[0].set_ylabel("Metric Value", fontsize=12)
    axes[0].set_title("Bias Metrics vs. Penalty", fontsize=14)
    axes[0].grid(True)
    axes[0].tick_params(axis="x", rotation=45)
    
    # Plotting disparate impact
    axes[1].plot(data.index, data["disparate_impact"], label="Disparate Impact", color='purple')

    # Add a horizontal dotted line at y=0.8 with label "Fair"
    axes[1].axhline(y=0.8, color='black', linestyle='--')
    axes[1].text(x=label_fair_index, y=label_fair_di_pos, s="Fair", color='black', ha='left', va='center', fontsize=10)
    axes[1].set_xlabel("Penalty", fontsize=12)
    axes[1].set_ylabel("Disparate Impact Value", fontsize=12)
    axes[1].set_title("Disparate Impact vs. Penalty", fontsize=14)
    axes[1].grid(True)
    axes[1].tick_params(axis="x", rotation=45)

    # Combine legends from both plots
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend(axes[1].get_legend_handles_labels()[0:1])  # Add disparate impact handle from second plot
    labels.extend(["Disparate Impact"])

    # Set the legend in the first subplot
    axes[0].legend(handles, labels, title="Metrics", loc=label_pos, fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_utility_metrics(data, label_pos):
    """
    Plot the utility metrics against the number of samples.
    """
    # Set up the color palette
    palette = sns.color_palette("husl", len(data.columns))

    # Set up the plot with one axis
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Plotting all the metrics from the dataframe
    for i, metric in enumerate(data.columns):
        ax.plot(data.index, data[metric], label=metric.replace('_', ' ').title(), color=palette[i])

    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Performance Metrics", fontsize=14)
    ax.grid(True)
    ax.tick_params(axis="x")

    # Set the legend
    ax.legend(title="Metrics", loc=label_pos, fontsize=10)

    plt.tight_layout()
    plt.show()
