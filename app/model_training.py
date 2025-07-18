import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Constants
FEATURE_COLUMNS = ["hPPI", "aPPI", "PPI_Diff"]
RANDOM_STATE = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)


class BettingModel:
    """Class to handle sports betting predictions using machine learning."""

    def __init__(self, feature_columns: List[str] = FEATURE_COLUMNS):
        """Initialize the betting model.

        Args:
            feature_columns: List of feature column names
        """
        self.feature_columns = feature_columns
        self.home_model = None
        self.draw_model = None
        self.away_model = None
        self.scaler = StandardScaler()

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

        from utils.odds_helpers import get_no_vig_odds_multiway

        return get_no_vig_odds_multiway(odds_list)

    def process_historical_data(
        self, file_path: str, season: list[str]
    ) -> pd.DataFrame:
        """Load and preprocess historical data.

        Args:
            file_path: Path to historical data CSV

        Returns:
            Processed DataFrame
        """
        logger.info(f"Loading historical data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Original data size: {df.shape[0]}")

        # Clean data
        df = df.dropna(subset=self.feature_columns + ["PSCH", "PSCD", "PSCA"])
        logger.info(f"Data size after removing NaNs: {df.shape[0]}")

        # Filter out matches with insufficient data
        df = df[df["Wk"] >= 10].copy()
        df = df[df["Season"].isin(season)]
        logger.info(f"Data size after filtering: {df.shape[0]}")
        logger.info(f"Using seasons: {df['Season'].unique()} for training")

        # Calculate fair odds
        df[["PSCH_fair_odds", "PSCD_fair_odds", "PSCA_fair_odds"]] = df.apply(
            lambda r: pd.Series(
                self.get_no_vig_odds_multiway([r["PSCH"], r["PSCD"], r["PSCA"]])
            ),
            axis=1,
        )

        # Data validation
        invalid_odds = df[
            (df["PSCH_fair_odds"] <= 1)
            | (df["PSCD_fair_odds"] <= 1)
            | (df["PSCA_fair_odds"] <= 1)
        ]
        if not invalid_odds.empty:
            logger.warning(f"Found {len(invalid_odds)} rows with invalid odds")
            df = df[
                (df["PSCH_fair_odds"] > 1)
                & (df["PSCD_fair_odds"] > 1)
                & (df["PSCA_fair_odds"] > 1)
            ]

        return df

    def train_models(
        self, df: pd.DataFrame, cv: int = 5, tune_hyperparams: bool = True
    ) -> None:
        """Train models for home, draw, and away outcomes.

        Args:
            df: Processed historical DataFrame
            cv: Number of cross-validation folds
            tune_hyperparams: Whether to perform hyperparameter tuning
        """
        # Extract features and targets
        X = df[self.feature_columns].values
        y_home = df["PSCH_fair_odds"].values
        y_draw = df["PSCD_fair_odds"].values
        y_away = df["PSCA_fair_odds"].values

        # Split data
        X_train, X_test, y_home_train, y_home_test = train_test_split(
            X, y_home, test_size=0.2, random_state=RANDOM_STATE
        )
        _, _, y_draw_train, y_draw_test = train_test_split(
            X, y_draw, test_size=0.2, random_state=RANDOM_STATE
        )
        _, _, y_away_train, y_away_test = train_test_split(
            X, y_away, test_size=0.2, random_state=RANDOM_STATE
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define parameter grid for hyperparameter tuning
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        models = {}
        targets = {
            "home": (y_home_train, y_home_test),
            "draw": (y_draw_train, y_draw_test),
            "away": (y_away_train, y_away_test),
        }

        for outcome_type, (y_train, y_test) in targets.items():
            logger.info(f"Training {outcome_type} model")

            if tune_hyperparams:
                # Hyperparameter tuning
                logger.info(
                    f"Performing hyperparameter tuning for {outcome_type} model"
                )
                base_model = RandomForestRegressor(random_state=RANDOM_STATE)
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=-1,
                    verbose=1,
                    scoring="r2",
                )
                grid_search.fit(X_train_scaled, y_train)

                # Get best model
                best_params = grid_search.best_params_
                logger.info(f"Best parameters for {outcome_type} model: {best_params}")
                model = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
            else:
                # Use default model
                model = RandomForestRegressor(
                    n_estimators=100, random_state=RANDOM_STATE
                )

            # Train on training data
            model.fit(X_train_scaled, y_train)

            # Evaluate on test data
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            logger.info(f"{outcome_type.capitalize()} model R² score: {r2:.4f}")
            logger.info(f"{outcome_type.capitalize()} model MSE: {mse:.4f}")

            # Cross-validation
            cv_scores = cross_val_score(
                model,
                X,
                df[f"PSC{outcome_type[0].upper()}_fair_odds"].values,
                cv=cv,
                scoring="r2",
            )
            logger.info(f"{outcome_type.capitalize()} model CV R² scores: {cv_scores}")
            logger.info(
                f"{outcome_type.capitalize()} model mean CV R² score: {cv_scores.mean():.4f}"
            )

            # Store model
            models[outcome_type] = model

            # Feature importance
            feature_importance = pd.DataFrame(
                {
                    "Feature": self.feature_columns,
                    "Importance": model.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)

            logger.info(
                f"{outcome_type.capitalize()} model feature importance:\n{feature_importance}"
            )

        # Retrain on all data
        logger.info("Retraining models on all data")
        X_all_scaled = self.scaler.fit_transform(X)

        self.home_model = models["home"]
        self.draw_model = models["draw"]
        self.away_model = models["away"]

        self.home_model.fit(X_all_scaled, y_home)
        self.draw_model.fit(X_all_scaled, y_draw)
        self.away_model.fit(X_all_scaled, y_away)

        # Save models
        joblib.dump(self.home_model, MODEL_DIR / "home_model.pkl")
        joblib.dump(self.draw_model, MODEL_DIR / "draw_model.pkl")
        joblib.dump(self.away_model, MODEL_DIR / "away_model.pkl")
        joblib.dump(self.scaler, MODEL_DIR / "scaler.pkl")


def main():
    """Main function to execute the betting model pipeline."""
    logger.info("Starting betting model training and prediction pipeline")

    # # Initialize model
    model = BettingModel(feature_columns=FEATURE_COLUMNS)

    # # Process historical data
    historical_data = model.process_historical_data(
        "historical_ppi_and_odds.csv", ["2022-2023", "2023-2024"]
    )

    # Train models
    model.train_models(historical_data, tune_hyperparams=True)

    logger.info("Betting model pipeline completed successfully")


if __name__ == "__main__":
    main()
