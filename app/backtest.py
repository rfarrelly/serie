from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class StakingMethod(ABC):
    def __init__(self, initial_bank: float = 1000.0):
        self.initial_bank = initial_bank
        self.current_bank = initial_bank
        self.history = []

    @abstractmethod
    def calculate_stake(self, odds: float, confidence: float = None) -> float:
        pass

    def update_bank(self, profit: float):
        self.current_bank += profit
        self.history.append(self.current_bank)

    def reset(self):
        self.current_bank = self.initial_bank
        self.history = []


class FixedStake(StakingMethod):
    def __init__(self, stake: float = 10.0, initial_bank: float = 1000.0):
        super().__init__(initial_bank)
        self.stake = stake

    def calculate_stake(self, odds: float, confidence: float = None) -> float:
        return min(self.stake, self.current_bank)


class PercentageStake(StakingMethod):
    def __init__(self, percentage: float = 1.0, initial_bank: float = 1000.0):
        super().__init__(initial_bank)
        self.percentage = max(0, min(100, percentage))

    def calculate_stake(self, odds: float, confidence: float = None) -> float:
        return self.current_bank * (self.percentage / 100)


class KellyStake(StakingMethod):
    def __init__(self, fraction: float = 1.0, initial_bank: float = 1000.0):
        super().__init__(initial_bank)
        self.fraction = max(0, min(1, fraction))

    def calculate_stake(self, odds: float, confidence: float = None) -> float:
        if confidence is None:
            confidence = 1 / odds
        b = odds - 1
        q = 1 - confidence
        if b * confidence > q:
            kelly_fraction = (b * confidence - q) / b
        else:
            kelly_fraction = 0
        stake_fraction = kelly_fraction * self.fraction
        return self.current_bank * stake_fraction


class DrawdownProtection(StakingMethod):
    def __init__(
        self,
        base_stake: float = 10.0,
        max_drawdown: float = 20.0,
        recovery_rate: float = 5.0,
        initial_bank: float = 1000.0,
    ):
        super().__init__(initial_bank)
        self.base_stake = base_stake
        self.max_drawdown = max_drawdown / 100.0
        self.recovery_rate = recovery_rate / 100.0
        self.peak_bank = initial_bank

    def calculate_stake(self, odds: float, confidence: float = None) -> float:
        if self.current_bank > self.peak_bank:
            self.peak_bank = self.current_bank
        if self.peak_bank > 0:
            current_drawdown = (self.peak_bank - self.current_bank) / self.peak_bank
        else:
            current_drawdown = 0
        if current_drawdown >= self.max_drawdown:
            reduction_factor = 0.0
        elif current_drawdown > 0:
            reduction_factor = 1.0 - (current_drawdown / self.max_drawdown)
        else:
            reduction_factor = 1.0
        stake = self.base_stake * reduction_factor
        return min(stake, self.current_bank)


class BettingStrategy(ABC):
    @abstractmethod
    def place_bet(self, row: pd.Series) -> pd.Series:
        pass

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_confidence(self, row: pd.Series) -> float:
        return None


class DoubleChanceStrategy(BettingStrategy):
    def __init__(self, ppi_threshold: float = 0.1):
        self.ppi_threshold = ppi_threshold

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        home_prob = 1 / df["PSH"]
        draw_prob = 1 / df["PSD"]
        away_prob = 1 / df["PSA"]
        dc_1x_prob = home_prob + draw_prob
        dc_x2_prob = draw_prob + away_prob
        df["dc_1x"] = round(1 / dc_1x_prob, 2)
        df["dc_x2"] = round(1 / dc_x2_prob, 2)
        return df

    def place_bet(self, row: pd.Series) -> pd.Series:
        row["Profit"] = 0.0  # Initialize as float
        row["acca_odds"] = 0.0  # Initialize as float
        row["Confidence"] = None
        row["BetType"] = "None"
        row["BetOdds"] = None
        if (row["dc_1x"] > row["dc_x2"]) and (row["PPI_Diff"] <= self.ppi_threshold):
            row["BetType"] = "DC_1X"
            row["BetOdds"] = row["dc_1x"]
            row["Confidence"] = max(
                0, min(1, 1 - (row["PPI_Diff"] / self.ppi_threshold))
            )
            if (row["FTR"] == "H") or (row["FTR"] == "D"):
                row["Profit"] = row["dc_1x"] - 1
                row["acca_odds"] = row["dc_1x"]
            else:
                row["Profit"] = -1
                row["acca_odds"] = "x"
        elif (row["dc_1x"] < row["dc_x2"]) and (row["PPI_Diff"] <= self.ppi_threshold):
            row["BetType"] = "DC_X2"
            row["BetOdds"] = row["dc_x2"]
            row["Confidence"] = max(
                0, min(1, 1 - (row["PPI_Diff"] / self.ppi_threshold))
            )
            if (row["FTR"] == "A") or (row["FTR"] == "D"):
                row["Profit"] = row["dc_x2"] - 1
                row["acca_odds"] = row["dc_x2"]
            else:
                row["Profit"] = -1
                row["acca_odds"] = "x"
        elif row["dc_1x"] == row["dc_x2"]:
            row["BetType"] = "Draw"
            row["BetOdds"] = row["PSD"]
            row["Confidence"] = max(
                0, min(1, 0.7 * (1 - (row["PPI_Diff"] / self.ppi_threshold)))
            )
            if row["FTR"] == "D":
                row["Profit"] = row["PSD"] - 1
                row["acca_odds"] = row["PSD"]
            else:
                row["Profit"] = -1
                row["acca_odds"] = "x"
        return row

    def get_confidence(self, row: pd.Series) -> float:
        return row.get("Confidence", 0.5)


class HighOddsStrategy(BettingStrategy):
    def __init__(self, odds_threshold: float = 2.5):
        self.odds_threshold = odds_threshold

    def place_bet(self, row: pd.Series) -> pd.Series:
        row["Profit"] = 0.0  # Initialize as float
        row["acca_odds"] = 0.0  # Initialize as float
        row["Confidence"] = None
        row["BetType"] = "None"
        row["BetOdds"] = None
        if row["PSH"] > self.odds_threshold:
            row["BetType"] = "Home"
            row["BetOdds"] = row["PSH"]
            row["Confidence"] = max(0.1, min(0.7, 1.5 / row["PSH"]))
            if row["FTR"] == "H":
                row["Profit"] = row["PSH"] - 1
                row["acca_odds"] = row["PSH"]
            else:
                row["Profit"] = -1
                row["acca_odds"] = "x"
        else:
            row["Profit"] = 0.0  # Use float instead of int
            row["acca_odds"] = 1.0  # Use float instead of int
        return row

    def get_confidence(self, row: pd.Series) -> float:
        return row.get("Confidence", 0.5)


class Backtester:
    def __init__(
        self,
        data_file: str,
        season: str = "2024-2025",
        min_week: int = 10,
        ppi_diff_threshold: float = 0.11,
        initial_bank: float = 1000.0,
    ):
        self.data_file = data_file
        self.season = season
        self.min_week = min_week
        self.ppi_diff_threshold = ppi_diff_threshold
        self.initial_bank = initial_bank
        self.raw_data = None
        self.backtest_data = None
        self.weekly_groups = []
        self.split_weekly_groups = []

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_file)
        backtest_df = df[df["Season"] == self.season].copy()
        backtest_df["Date"] = pd.to_datetime(backtest_df["Date"])
        backtest_df = backtest_df.sort_values("Date")
        backtest_df = backtest_df.rename(
            columns={"PSCH": "PSH", "PSCD": "PSD", "PSCA": "PSA"}
        )
        backtest_df["FTR"] = backtest_df.apply(
            lambda x: (
                "H" if x["FTHG"] > x["FTAG"] else "D" if x["FTHG"] == x["FTAG"] else "A"
            ),
            axis="columns",
        )
        self.raw_data = backtest_df
        return backtest_df

    def group_by_week(self) -> List[pd.DataFrame]:
        if self.raw_data is None:
            self.load_data()

        start_date = self.raw_data["Date"].min()
        self.raw_data["WeekGroup"] = (
            (self.raw_data["Date"] - start_date).dt.days // 7
        ) + 1

        weekly_groups = [group for _, group in self.raw_data.groupby("WeekGroup")]

        weekly_groups_filtered = []
        for wg_df in weekly_groups:
            filtered_df = wg_df[wg_df["Wk"] > self.min_week]
            filtered_df = filtered_df[filtered_df["PPI_Diff"] < self.ppi_diff_threshold]

            if not filtered_df.empty:
                weekly_groups_filtered.append(filtered_df)

        self.weekly_groups = weekly_groups_filtered
        return weekly_groups_filtered

    def split_by_days(self) -> List[pd.DataFrame]:
        if not self.weekly_groups:
            self.group_by_week()
        first_half_days = {"Fri", "Sat", "Sun", "Mon"}
        second_half_days = {"Tue", "Wed", "Thu"}
        split_weekly_groups = []
        for group in self.weekly_groups:
            first_half_df = (
                group[group["Day"].isin(first_half_days)]
                .copy()
                .sort_values("PPI_Diff")
                .reset_index(drop=True)
            )
            second_half_df = (
                group[group["Day"].isin(second_half_days)]
                .copy()
                .sort_values("PPI_Diff")
                .reset_index(drop=True)
            )
            if not first_half_df.empty and first_half_df.shape[0] > 3:
                split_weekly_groups.append(first_half_df)
            if not second_half_df.empty and second_half_df.shape[0] > 3:
                split_weekly_groups.append(second_half_df)
        self.split_weekly_groups = split_weekly_groups
        return split_weekly_groups

    def run_backtest(
        self,
        strategy: BettingStrategy,
        staking_method: StakingMethod = None,
        group_selection: str = "split",
        picks_per_group: int = 4,
        acca_bet: bool = True,
        acca_stake: float = 10.0,
    ) -> Dict[str, Union[pd.DataFrame, List[Dict[str, float]]]]:
        if group_selection == "split" and not self.split_weekly_groups:
            self.split_by_days()
            groups_to_test = self.split_weekly_groups
        elif group_selection == "weekly" and not self.weekly_groups:
            self.group_by_week()
            groups_to_test = self.weekly_groups
        else:
            groups_to_test = (
                self.split_weekly_groups
                if group_selection == "split"
                else self.weekly_groups
            )
        if staking_method is None:
            staking_method = FixedStake(stake=1.0, initial_bank=self.initial_bank)
        staking_method.reset()
        acca_profits = []
        group_results = []
        bet_results = []
        for group in groups_to_test:
            group = strategy.prepare_data(group.copy())
            # Initialize columns with appropriate data types
            group["Profit"] = 0.0  # Use float
            group["acca_odds"] = 0.0  # Use float
            group["cumprofit"] = 0.0  # Use float
            group["Stake"] = 0.0  # Use float
            group["ActualProfit"] = 0.0  # Use float
            group = group.apply(strategy.place_bet, axis="columns")
            top_picks = group.nsmallest(picks_per_group, "PPI_Diff")
            for idx, row in top_picks.iterrows():
                if row["BetType"] != "None" and row["BetOdds"] is not None:
                    confidence = strategy.get_confidence(row)
                    stake = staking_method.calculate_stake(row["BetOdds"], confidence)
                    actual_profit = stake * row["Profit"]
                    staking_method.update_bank(actual_profit)
                    # Convert idx to integer as it might be a Label
                    int_idx = (
                        group.index.get_loc(idx) if not isinstance(idx, int) else idx
                    )
                    group.at[idx, "Stake"] = stake
                    group.at[idx, "ActualProfit"] = actual_profit
                    bet_results.append(
                        {
                            "WeekGroup": group["WeekGroup"].values[0],
                            "Date": row["Date"],
                            "Match": f"{row['Home']} vs {row['Away']}",
                            "BetType": row["BetType"],
                            "Odds": row["BetOdds"],
                            "Stake": stake,
                            "Profit": actual_profit,
                            "BankBalance": staking_method.current_bank,
                        }
                    )
            if acca_bet:
                if "x" in top_picks["acca_odds"].values:
                    acca_profit = -acca_stake
                else:
                    valid_odds = pd.to_numeric(top_picks["acca_odds"], errors="coerce")
                    valid_odds = valid_odds[~np.isnan(valid_odds)]
                    if not valid_odds.empty:
                        acca_odds_product = valid_odds.product()
                        acca_profit = (acca_stake * acca_odds_product) - acca_stake
                    else:
                        acca_profit = 0.0
                acca_profits.append(acca_profit)
            group_profit = group["ActualProfit"].sum()
            group_results.append(
                {
                    "WeekGroup": group["WeekGroup"].values[0],
                    "Day": (
                        group["Day"].values[0]
                        if group["Day"].nunique() == 1
                        else "Mixed"
                    ),
                    "Profit": group_profit,
                    "NumMatches": len(top_picks),
                    "NumBets": len(top_picks[top_picks["BetType"] != "None"]),
                    "BankBalance": staking_method.current_bank,
                }
            )
        results_df = pd.DataFrame(group_results)
        if not results_df.empty:
            results_df["cumprofit"] = results_df["Profit"].cumsum()
        bet_results_df = pd.DataFrame(bet_results)
        return {
            "results": results_df,
            "bet_results": bet_results_df,
            "acca_profits": acca_profits,
            "total_profit": results_df["Profit"].sum() if not results_df.empty else 0,
            "avg_profit_per_group": (
                results_df["Profit"].mean() if not results_df.empty else 0
            ),
            "num_groups": len(results_df),
            "total_acca_profit": sum(acca_profits) if acca_profits else 0,
            "final_bank": staking_method.current_bank,
            "bank_growth": (
                ((staking_method.current_bank / staking_method.initial_bank) - 1) * 100
                if staking_method.initial_bank > 0
                else 0
            ),
            "bank_history": staking_method.history,
        }

    def plot_results(self, results: Dict[str, Any], title: str = "Backtest Results"):
        if "results" not in results or results["results"].empty:
            print("No results to plot")
            return
        plt.figure(figsize=(12, 16))
        plt.subplot(3, 1, 1)
        plt.plot(results["bank_history"])
        plt.title(f"{title} - Bank Balance History")
        plt.xlabel("Bet Number")
        plt.ylabel("Bank Balance")
        plt.grid(True)
        plt.subplot(3, 1, 2)
        results["results"].plot(y="cumprofit", x="WeekGroup", kind="line", ax=plt.gca())
        plt.title(f"{title} - Cumulative Profit")
        plt.grid(True)
        plt.subplot(3, 1, 3)
        results["results"].plot(y="Profit", x="WeekGroup", kind="bar", ax=plt.gca())
        plt.title(f"{title} - Weekly Profits")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def compare_strategies(
        self,
        strategies_results: Dict[str, Dict[str, Any]],
        plot_type: str = "bank_balance",
    ):
        plt.figure(figsize=(12, 8))
        if plot_type == "bank_balance":
            for strategy_name, results in strategies_results.items():
                if "bank_history" in results and results["bank_history"]:
                    plt.plot(results["bank_history"], label=strategy_name)
            plt.title("Bank Balance Comparison")
            plt.xlabel("Bet Number")
            plt.ylabel("Bank Balance")
        elif plot_type == "cumulative":
            for strategy_name, results in strategies_results.items():
                if "results" in results and not results["results"].empty:
                    results["results"].plot(
                        y="cumprofit", x="WeekGroup", label=strategy_name, ax=plt.gca()
                    )
            plt.title("Cumulative Profit Comparison")
            plt.xlabel("Week Group")
            plt.ylabel("Cumulative Profit")
        elif plot_type == "roi":
            for strategy_name, results in strategies_results.items():
                if "results" in results and not results["results"].empty:
                    initial_bank = (
                        results["results"]["BankBalance"].iloc[0]
                        - results["results"]["Profit"].iloc[0]
                    )
                    roi_series = (
                        (results["results"]["BankBalance"] / initial_bank) - 1
                    ) * 100
                    plt.plot(
                        results["results"]["WeekGroup"], roi_series, label=strategy_name
                    )
            plt.title("Return on Investment (ROI) Comparison")
            plt.xlabel("Week Group")
            plt.ylabel("ROI (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_detailed_statistics(self, results: Dict[str, Any]) -> pd.DataFrame:
        stats = {}
        if "bet_results" in results and not results["bet_results"].empty:
            bet_df = results["bet_results"]
            stats["Total Bets"] = len(bet_df)
            winning_bets = bet_df[bet_df["Profit"] > 0]
            stats["Winning Bets"] = len(winning_bets)
            stats["Win Rate"] = (
                len(winning_bets) / len(bet_df) if len(bet_df) > 0 else 0
            )
            stats["Total Profit"] = bet_df["Profit"].sum()
            stats["Average Profit per Bet"] = bet_df["Profit"].mean()
            stats["Profit Standard Deviation"] = bet_df["Profit"].std()
            stats["Maximum Profit"] = bet_df["Profit"].max()
            stats["Maximum Loss"] = bet_df["Profit"].min()
            stats["Initial Bank"] = (
                bet_df["BankBalance"].iloc[0] - bet_df["Profit"].iloc[0]
            )
            stats["Final Bank"] = bet_df["BankBalance"].iloc[-1]
            stats["ROI"] = ((stats["Final Bank"] / stats["Initial Bank"]) - 1) * 100
            bank_series = np.array(
                [stats["Initial Bank"]] + list(bet_df["BankBalance"])
            )
            running_max = np.maximum.accumulate(bank_series)
            drawdown = (running_max - bank_series) / running_max
            stats["Maximum Drawdown"] = drawdown.max() * 100
            if bet_df["Profit"].std() > 0:
                stats["Sharpe Ratio"] = (
                    bet_df["Profit"].mean() / bet_df["Profit"].std() * np.sqrt(252)
                )
            else:
                stats["Sharpe Ratio"] = float("nan")
            if "BetType" in bet_df.columns:
                bet_type_stats = {}
                for bet_type in bet_df["BetType"].unique():
                    type_df = bet_df[bet_df["BetType"] == bet_type]
                    type_wins = type_df[type_df["Profit"] > 0]
                    bet_type_stats[f"{bet_type}_Count"] = len(type_df)
                    bet_type_stats[f"{bet_type}_Win_Rate"] = (
                        len(type_wins) / len(type_df) if len(type_df) > 0 else 0
                    )
                    bet_type_stats[f"{bet_type}_Profit"] = type_df["Profit"].sum()
                stats.update(bet_type_stats)
        if "acca_profits" in results and results["acca_profits"]:
            acca_profits = results["acca_profits"]
            stats["Acca Bets"] = len(acca_profits)
            stats["Winning Acca Bets"] = sum(1 for p in acca_profits if p > 0)
            stats["Acca Win Rate"] = (
                stats["Winning Acca Bets"] / stats["Acca Bets"]
                if stats["Acca Bets"] > 0
                else 0
            )
            stats["Total Acca Profit"] = sum(acca_profits)
            stats["Average Acca Profit"] = (
                sum(acca_profits) / len(acca_profits) if len(acca_profits) > 0 else 0
            )
        return pd.DataFrame(list(stats.items()), columns=["Statistic", "Value"])


def main():
    backtester = Backtester(
        data_file="historical_rpi_and_odds.csv",
        season="2024-2025",
        min_week=10,
        ppi_diff_threshold=0.11,
        initial_bank=1000.0,
    )
    dc_strategy = DoubleChanceStrategy(ppi_threshold=0.1)
    high_odds_strategy = HighOddsStrategy(odds_threshold=3.0)
    fixed_stake = FixedStake(stake=10.0, initial_bank=1000.0)
    percentage_stake = PercentageStake(percentage=2.0, initial_bank=1000.0)
    kelly_stake = KellyStake(fraction=0.5, initial_bank=1000.0)
    drawdown_prot = DrawdownProtection(
        base_stake=10.0, max_drawdown=20.0, initial_bank=1000.0
    )
    results = {}
    print("Running Double Chance Strategy with Fixed Stake...")
    results["DC_Fixed"] = backtester.run_backtest(
        strategy=dc_strategy,
        staking_method=fixed_stake,
        group_selection="split",
        picks_per_group=4,
        acca_bet=True,
    )
    print("Running Double Chance Strategy with Kelly Criterion...")
    results["DC_Kelly"] = backtester.run_backtest(
        strategy=dc_strategy,
        staking_method=kelly_stake,
        group_selection="split",
        picks_per_group=4,
        acca_bet=True,
    )
    print("Running High Odds Strategy with Percentage Stake...")
    results["HO_Percentage"] = backtester.run_backtest(
        strategy=high_odds_strategy,
        staking_method=percentage_stake,
        group_selection="split",
        picks_per_group=4,
        acca_bet=True,
    )
    print("Running High Odds Strategy with Drawdown Protection...")
    results["HO_Drawdown"] = backtester.run_backtest(
        strategy=high_odds_strategy,
        staking_method=drawdown_prot,
        group_selection="split",
        picks_per_group=4,
        acca_bet=True,
    )
    for name, result in results.items():
        print(f"\n{name} Strategy Results:")
        print(f"Initial Bank: £{fixed_stake.initial_bank:.2f}")
        print(f"Final Bank: £{result['final_bank']:.2f}")
        print(f"Bank Growth: {result['bank_growth']:.2f}%")
        print(f"Total Profit: £{result['total_profit']:.2f}")
        print(f"Total Accumulator Profit: £{result['total_acca_profit']:.2f}")
        print(f"Number of Groups: {result['num_groups']}")
        print(f"Average Profit per Group: £{result['avg_profit_per_group']:.2f}")
    detailed_stats = backtester.get_detailed_statistics(results["DC_Kelly"])
    print("\nDetailed Statistics for Double Chance with Kelly Staking:")
    print(detailed_stats)
    backtester.plot_results(results["DC_Fixed"], "Double Chance with Fixed Stake")
    backtester.plot_results(results["DC_Kelly"], "Double Chance with Kelly Criterion")
    backtester.compare_strategies(results, plot_type="bank_balance")
    backtester.compare_strategies(results, plot_type="roi")


if __name__ == "__main__":
    main()
