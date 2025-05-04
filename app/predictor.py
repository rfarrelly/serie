import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
FEATURE_COLUMNS = ["hPPI", "aPPI", "PPI_Diff"]
VALUE_THRESHOLD = 1.15


class BettingPredictor:
    """Class to load saved models and make predictions on new data."""

    def __init__(
        self, model_dir: Path = MODEL_DIR, feature_columns: List[str] = FEATURE_COLUMNS
    ):
        """Initialize the predictor by loading saved models.

        Args:
            model_dir: Directory containing saved models
            feature_columns: Feature column names used by the models
        """
        self.model_dir = model_dir
        self.feature_columns = feature_columns
        self.home_model = None
        self.draw_model = None
        self.away_model = None
        self.scaler = None

        # Load models
        self.load_models()

    def load_models(self) -> None:
        """Load trained models and scaler from files."""
        try:
            logger.info(f"Loading models from {self.model_dir}")
            self.home_model = joblib.load(self.model_dir / "home_model.pkl")
            self.draw_model = joblib.load(self.model_dir / "draw_model.pkl")
            self.away_model = joblib.load(self.model_dir / "away_model.pkl")
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            logger.info("Models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Error loading models: {e}")
            raise

    @staticmethod
    def convert_odds_to_probability(odds: np.ndarray) -> np.ndarray:
        """Convert decimal odds to probability.

        Args:
            odds: Array of decimal odds

        Returns:
            Array of probabilities
        """
        return 1 / odds

    @staticmethod
    def get_no_vig_odds_multiway(odds_list: List[float]) -> List[float]:
        """Remove the vig from a set of odds.

        Args:
            odds_list: List of decimal odds

        Returns:
            List of fair odds with no vig
        """
        try:
            from utils.odds_helpers import get_no_vig_odds_multiway

            return get_no_vig_odds_multiway(odds_list)
        except ImportError:
            # Fallback implementation if the utils module is not available
            probs = [1 / odds for odds in odds_list]
            total_prob = sum(probs)
            fair_probs = [prob / total_prob for prob in probs]
            fair_odds = [1 / prob for prob in fair_probs]
            return fair_odds

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare input data for prediction.

        Args:
            data: DataFrame containing match data

        Returns:
            Processed DataFrame ready for prediction
        """
        data = data.dropna(subset=self.feature_columns + ["PSH", "PSD", "PSA"])
        # Ensure required columns exist
        required_columns = self.feature_columns + ["PSH", "PSD", "PSA"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Copy to avoid modifying original data
        df = data.copy()

        # Calculate fair odds
        df[["PSH_fair_odds", "PSD_fair_odds", "PSA_fair_odds"]] = df.apply(
            lambda r: pd.Series(
                self.get_no_vig_odds_multiway([r["PSH"], r["PSD"], r["PSA"]])
            ),
            axis=1,
        )

        # Convert to probabilities
        df["PSH_fair_prob"] = self.convert_odds_to_probability(df["PSH_fair_odds"])
        df["PSD_fair_prob"] = self.convert_odds_to_probability(df["PSD_fair_odds"])
        df["PSA_fair_prob"] = self.convert_odds_to_probability(df["PSA_fair_odds"])

        return df

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for new matches.

        Args:
            data: DataFrame containing match data with required columns

        Returns:
            DataFrame with predictions and value assessments
        """
        # Check if models are loaded
        if any(
            model is None
            for model in [
                self.home_model,
                self.draw_model,
                self.away_model,
                self.scaler,
            ]
        ):
            raise ValueError("Models must be loaded before making predictions")

        # Prepare data
        df = self.prepare_data(data)

        # Extract features
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # Make predictions
        df["pred_PSH"] = np.round(self.home_model.predict(X_scaled), 3)
        df["pred_PSD"] = np.round(self.draw_model.predict(X_scaled), 3)
        df["pred_PSA"] = np.round(self.away_model.predict(X_scaled), 3)

        # Data validation - ensure all predictions are valid odds (>1)
        df["pred_PSH"] = np.maximum(df["pred_PSH"], 1.01)
        df["pred_PSD"] = np.maximum(df["pred_PSD"], 1.01)
        df["pred_PSA"] = np.maximum(df["pred_PSA"], 1.01)

        # Convert to probabilities
        df["pred_PSH_prob"] = self.convert_odds_to_probability(df["pred_PSH"])
        df["pred_PSD_prob"] = self.convert_odds_to_probability(df["pred_PSD"])
        df["pred_PSA_prob"] = self.convert_odds_to_probability(df["pred_PSA"])

        # Normalize probabilities
        prob_sum = df["pred_PSH_prob"] + df["pred_PSD_prob"] + df["pred_PSA_prob"]
        df["pred_PSH_prob_norm"] = np.round(df["pred_PSH_prob"] / prob_sum, 3)
        df["pred_PSD_prob_norm"] = np.round(df["pred_PSD_prob"] / prob_sum, 3)
        df["pred_PSA_prob_norm"] = np.round(df["pred_PSA_prob"] / prob_sum, 3)

        # Calculate value ratios
        df["PSH_value"] = np.round(df["pred_PSH_prob_norm"] / df["PSH_fair_prob"], 3)
        df["PSD_value"] = np.round(df["pred_PSD_prob_norm"] / df["PSD_fair_prob"], 3)
        df["PSA_value"] = np.round(df["pred_PSA_prob_norm"] / df["PSA_fair_prob"], 3)

        return df

    def identify_value_bets(
        self, predictions: pd.DataFrame, threshold: float = VALUE_THRESHOLD
    ) -> Dict[str, pd.DataFrame]:
        """Identify value betting opportunities.

        Args:
            predictions: DataFrame with model predictions
            threshold: Value ratio threshold

        Returns:
            Dictionary of DataFrames with value bets for each outcome
        """
        home_value_bets = predictions[predictions["PSH_value"] > threshold]
        draw_value_bets = predictions[predictions["PSD_value"] > threshold]
        away_value_bets = predictions[predictions["PSA_value"] > threshold]

        all_value_bets = predictions[
            (predictions["PSH_value"] > threshold)
            | (predictions["PSD_value"] > threshold)
            | (predictions["PSA_value"] > threshold)
        ]

        value_bets = {
            "home": home_value_bets,
            "draw": draw_value_bets,
            "away": away_value_bets,
            "all": all_value_bets,
        }

        # Log found value bets
        for outcome, bets in value_bets.items():
            if outcome != "all":
                logger.info(f"\n{outcome.capitalize()} Value Bets ({len(bets)}):")
                if not bets.empty:
                    odds_col = f"PS{outcome[0].upper()}"
                    pred_col = f"pred_PS{outcome[0].upper()}"
                    value_col = f"PS{outcome[0].upper()}_value"
                    display_cols = ["Home", "Away", odds_col, pred_col, value_col]
                    display_cols = [col for col in display_cols if col in bets.columns]
                    logger.info(bets[display_cols])

        return value_bets

    def save_results(
        self,
        value_bets: Dict[str, pd.DataFrame],
        output_file: str = "new_value_bets.csv",
    ) -> None:
        """Save value betting results to CSV.

        Args:
            value_bets: Dictionary of value bets
            output_file: Filename to save results
        """
        # Create results directory if it doesn't exist
        RESULTS_DIR.mkdir(exist_ok=True)

        # Save all value bets
        if not value_bets["all"].empty:
            output_columns = [
                "Date",
                "Time",
                "League",
                "Home",
                "Away",
                "hPPI",
                "aPPI",
                "PPI_Diff",
                "PSH",
                "PSD",
                "PSA",
                "pred_PSH",
                "pred_PSD",
                "pred_PSA",
                "PSH_value",
                "PSD_value",
                "PSA_value",
            ]

            # Only include columns that exist in the DataFrame
            available_columns = [
                col for col in output_columns if col in value_bets["all"].columns
            ]

            # Add PPI_Diff if it doesn't exist but component columns do
            if (
                "PPI_Diff" not in value_bets["all"].columns
                and "hPPI" in value_bets["all"].columns
                and "aPPI" in value_bets["all"].columns
            ):
                value_bets["all"]["PPI_Diff"] = (
                    value_bets["all"]["hPPI"] - value_bets["all"]["aPPI"]
                )
                available_columns.append("PPI_Diff")

            value_bets["all"][available_columns].to_csv(
                RESULTS_DIR / output_file, index=False
            )
            logger.info(f"Value bets saved to {RESULTS_DIR / output_file}")

    def visualize_results(
        self, predictions: pd.DataFrame, threshold: float = VALUE_THRESHOLD
    ) -> None:
        """Create visualizations of model results.

        Args:
            predictions: DataFrame with model predictions
            threshold: Value ratio threshold
        """
        # Set the style for better visualizations
        sns.set(style="whitegrid")

        # Create plots directory if it doesn't exist
        plots_dir = RESULTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Value vs. Odds plots
        plt.figure(figsize=(18, 12))

        outcome_types = [("Home", "PSH"), ("Draw", "PSD"), ("Away", "PSA")]

        for i, (outcome_name, outcome_code) in enumerate(outcome_types, 1):
            plt.subplot(2, 3, i)
            sns.scatterplot(
                data=predictions,
                x=outcome_code,
                y=f"{outcome_code}_value",
                hue="League",
                alpha=0.7,
            )
            plt.axhline(y=threshold, color="r", linestyle="--")
            plt.title(f"{outcome_name} Win Value")
            plt.xlabel("Bookmaker Odds")
            plt.ylabel("Value Ratio")
            plt.legend().remove()  # Remove legend for cleaner look

            # Add value threshold annotation
            plt.annotate(
                f"Value Threshold: {threshold}",
                xy=(0.05, threshold + 0.02),
                xycoords=("axes fraction", "data"),
                color="r",
            )

        # Add common legend at the bottom
        plt.subplot(2, 3, 6)
        plt.axis("off")
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc="center", title="League")

        # Add distribution of predicted vs. bookmaker odds
        for i, (outcome_name, outcome_code) in enumerate(outcome_types, 4):
            if i > 5:  # Only show first three in this layout
                continue

            plt.subplot(2, 3, i)
            ax = plt.gca()
            scatter = ax.scatter(
                predictions[outcome_code],
                predictions[f"pred_{outcome_code}"],
                c=predictions[f"{outcome_code}_value"],
                cmap="viridis",
                alpha=0.7,
            )

            # Plot the diagonal line (y=x)
            min_odds = min(
                predictions[outcome_code].min(),
                predictions[f"pred_{outcome_code}"].min(),
            )
            max_odds = max(
                predictions[outcome_code].max(),
                predictions[f"pred_{outcome_code}"].max(),
            )
            plt.plot([min_odds, max_odds], [min_odds, max_odds], "k--", alpha=0.5)

            plt.title(f"{outcome_name}: Predicted vs. Bookmaker Odds")
            plt.xlabel("Bookmaker Odds")
            plt.ylabel("Predicted Odds")
            plt.colorbar(scatter, label="Value Ratio")

        plt.tight_layout()
        plt.savefig(plots_dir / "value_analysis.png", dpi=300)
        plt.close()

        # Feature importance visualization
        plt.figure(figsize=(15, 5))

        for i, (outcome_name, model) in enumerate(
            [
                ("Home", self.home_model),
                ("Draw", self.draw_model),
                ("Away", self.away_model),
            ],
            1,
        ):
            plt.subplot(1, 3, i)
            feature_importance = pd.DataFrame(
                {
                    "Feature": self.feature_columns,
                    "Importance": model.feature_importances_,
                }
            ).sort_values("Importance", ascending=True)

            sns.barplot(
                x="Importance",
                y="Feature",
                data=feature_importance,
                palette="viridis",
                hue="Feature",
            )
            plt.title(f"{outcome_name} Model - Feature Importance")
            plt.tight_layout()

        plt.savefig(plots_dir / "feature_importance.png", dpi=300)
        plt.close()

        logger.info(f"Visualizations saved to {plots_dir}")


def example_usage():
    """Example of how to use the BettingPredictor."""
    # 1. Load saved models
    predictor = BettingPredictor()

    # 2. Load new match data
    # This could be from a CSV file, API, or manually created DataFrame
    try:
        new_matches = pd.read_csv("latest_rpi_and_odds.csv")
        logger.info(f"Loaded {len(new_matches)} new matches for prediction")
    except FileNotFoundError:
        # Create a sample DataFrame for demonstration
        logger.info("Creating sample match data for demonstration")
        new_matches = pd.DataFrame(
            {
                "Date": ["2025-05-10", "2025-05-10"],
                "Time": ["15:00", "17:30"],
                "League": ["Premier League", "La Liga"],
                "Home": ["Liverpool", "Barcelona"],
                "Away": ["Manchester City", "Real Madrid"],
                "hPPI": [85.2, 88.7],
                "aPPI": [83.7, 87.9],
                "PSH": [2.50, 2.30],
                "PSD": [3.40, 3.50],
                "PSA": [3.00, 3.10],
            }
        )

    # 3. Make predictions
    predictions = predictor.predict(new_matches)

    # 4. Identify value bets
    value_bets = predictor.identify_value_bets(predictions)

    # 5. Save results
    predictor.save_results(value_bets)

    predictor.visualize_results(predictions)

    # 6. Print summary
    print("\nPrediction Summary:")
    print("-" * 80)
    print(f"Total matches analyzed: {len(predictions)}")
    print(f"Home value bets found: {len(value_bets['home'])}")
    print(f"Draw value bets found: {len(value_bets['draw'])}")
    print(f"Away value bets found: {len(value_bets['away'])}")
    print(f"Total value bets found: {len(value_bets['all'])}")
    print("-" * 80)

    # 7. Return for further analysis if needed
    return predictions, value_bets


def interactive_mode():
    """Interactive mode for entering match details manually."""
    predictor = BettingPredictor()

    print("\n=== Sports Betting Value Finder ===")
    print("Enter match details to get predictions and value analysis")

    matches = []
    while True:
        print("\nEnter match details (or press Enter to finish):")

        # Basic match info
        date = input("Date (YYYY-MM-DD): ")
        if not date:
            break

        home_team = input("Home Team: ")
        away_team = input("Away Team: ")

        # Performance metrics
        try:
            home_ppi = float(input("Home Team PPI: "))
            away_ppi = float(input("Away Team PPI: "))
        except ValueError:
            print("Error: PPI values must be numbers")
            continue

        # Odds
        try:
            home_odds = float(input("Home Win Odds: "))
            draw_odds = float(input("Draw Odds: "))
            away_odds = float(input("Away Win Odds: "))
        except ValueError:
            print("Error: Odds must be numbers")
            continue

        # Add match to list
        matches.append(
            {
                "Date": date,
                "Time": "00:00",  # Default value
                "League": "Unknown",  # Default value
                "Home": home_team,
                "Away": away_team,
                "hPPI": home_ppi,
                "aPPI": away_ppi,
                "PSH": home_odds,
                "PSD": draw_odds,
                "PSA": away_odds,
            }
        )

        print(f"Added match: {home_team} vs {away_team}")

    if not matches:
        print("No matches entered. Exiting.")
        return

    # Create DataFrame and make predictions
    new_matches = pd.DataFrame(matches)
    predictions = predictor.predict(new_matches)
    value_bets = predictor.identify_value_bets(predictions)

    # Display results
    print("\n=== Results ===")
    for outcome in ["home", "draw", "away"]:
        if not value_bets[outcome].empty:
            print(f"\n{outcome.capitalize()} Value Bets:")
            odds_col = f"PS{outcome[0].upper()}"
            pred_col = f"pred_PS{outcome[0].upper()}"
            value_col = f"PS{outcome[0].upper()}_value"
            print(value_bets[outcome][["Home", "Away", odds_col, pred_col, value_col]])

    # Save results
    predictor.save_results(value_bets, "manual_value_bets.csv")
    print(f"\nResults saved to {RESULTS_DIR / 'manual_value_bets.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Use trained betting models for prediction"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode to manually enter matches",
    )
    parser.add_argument(
        "--file", "-f", type=str, help="Path to CSV file with match data"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.file:
        # Load custom file
        predictor = BettingPredictor()
        try:
            new_matches = pd.read_csv(args.file)
            print(f"Loaded {len(new_matches)} matches from {args.file}")
            predictions = predictor.predict(new_matches)
            value_bets = predictor.identify_value_bets(predictions)
            predictor.save_results(value_bets, f"value_bets_{Path(args.file).stem}.csv")
        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        # Run example with default data
        example_usage()
