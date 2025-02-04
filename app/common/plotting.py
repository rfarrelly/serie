import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_compare_team_rolling_stats(
    dataframes: list[pd.DataFrame],
    teams: list[str],
    target_stat: str,
    window: int = 1,
    show=True,
    save_path=None,
    filename=None,
):
    """
    Plots rolling statistics comparison for multiple teams.

    Parameters:
        dataframes (list[pd.DataFrame]): A list of DataFrames, each corresponding to a team.
        teams (list[str]): A list of team names corresponding to the DataFrames.
        target_stat (str): The statistical column to plot.
        window (int): The rolling window size.
        show (bool): Show the plot in a new window.
        save_path: The folder to save the plot.
        filename: The name of the plot.

    Returns:
        None
    """
    if len(dataframes) != len(teams):
        raise ValueError("The number of dataframes must match the number of teams.")

    plt.figure(figsize=(12, 8))

    # Iterate over each team and its corresponding DataFrame
    for df, team in zip(dataframes, teams):
        stat = df[target_stat]
        plt.plot(
            df.index,
            stat.rolling(window=window).mean(),
            label=team,
            marker="o",
            markersize=4,
        )

        if not stat.isna().all():
            last_index = stat.last_valid_index()
            last_value = stat[last_index]
            plt.text(
                last_index,
                last_value,
                f"{last_value:.2f}",
                fontsize=10,
                ha="left",
                va="bottom",
            )

    plt.xlabel("Index")
    plt.ylabel(target_stat)
    plt.title(f"{target_stat} Comparison")
    plt.legend()
    plt.grid()

    os.makedirs(save_path, exist_ok=True)

    # Save the figure
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, bbox_inches="tight")
    print(f"File '{filename}' downloaded and saved to '{save_path}'")
    if show:
        plt.show()
    else:
        plt.close()
