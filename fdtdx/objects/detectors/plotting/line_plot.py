import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_line_over_time(
    arr: np.ndarray,
    time_steps: list[float],  # in seconds
    metric_name: str,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(x=time_steps, y=arr, ax=ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(metric_name)
    return fig