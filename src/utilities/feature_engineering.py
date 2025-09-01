import holidays
import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from src.config import app_config


class FeatureEngineer:
    def __init__(self) -> None:
        self.feature_config = app_config.features
        self.validation_config = app_config.validation

    def create_date_features(self, df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """
        Create date features from a date column.

        Parameters
        ----------
        df : pl.DataFrame
            Dataframe to create date features from.
        date_col : str, optional
            Name of the date column, by default "date".

        Returns
        -------
        pl.DataFrame
            Dataframe with the created date features.

        Notes
        -----
        Polars by default uses a non-zero-based index for date components.
        """
        date_features: list[str] = self.feature_config.date_features
        df = df.clone()
        us_holidays = holidays.UnitedStates()  # type: ignore

        # Convert to datetime
        try:
            df = df.with_columns(pl.col("date").cast(pl.Date).alias("date"))
        except Exception as e:
            print(f"Error converting {date_col} to datetime: {e}")
            pass

        if "date" in date_features:
            return df.with_columns(
                pl.col("date").dt.day().alias("day"),
                pl.col("date").dt.month().alias("month"),
                pl.col("date").dt.year().alias("year"),
                (pl.col("date").dt.weekday() - 1).alias("day_of_week"),
                pl.col("date").dt.weekday().alias("day_of_week"),
                pl.col("date").dt.quarter().alias("quarter"),
                pl.col("date").dt.week().alias("week_of_year"),
                ((pl.col("date").dt.weekday() - 1) >= 5).alias("is_weekend").cast(pl.Int8),
                pl.col("date").map_elements(lambda x: x in us_holidays).alias("is_holiday").cast(pl.Int8),
            )
        return df

    def create_lag_features(self, df: pl.DataFrame, target_col: str, group_cols: list[str] | None = None) -> pl.DataFrame:
        """
        Create lag features from a target column.

        Parameters
        ----------
        df : pl.DataFrame
            Dataframe to create lag features from.
        target_col : str
            Name of the target column to create lag features from.
        group_cols : list[str] | None, optional
            Columns to group by when creating lag features, by default None.

        Returns
        -------
        pl.DataFrame
            Dataframe with the created lag features.
        """
        df = df.clone()
        lag_values = self.feature_config.lag_features

        if group_cols:
            for lag in lag_values:
                df = df.with_columns(pl.col(target_col).shift(lag).over(group_cols).alias(f"{target_col}_lag_{lag}"))
        else:
            for lag in lag_values:
                df = df.with_columns(pl.col(target_col).shift(lag).alias(f"{target_col}_lag_{lag}"))

        print(f"Created {len(lag_values)} lag features")
        return df

    def create_rolling_features(
        self, df: pl.DataFrame, target_col: str, group_cols: list[str] | None = None
    ) -> pl.DataFrame:
        """
        Create rolling features from a target column.

        Parameters
        ----------
        df : pl.DataFrame
            Dataframe to create rolling features from.
        target_col : str
            Name of the target column to create rolling features from.
        group_cols : list[str] | None, optional
            Columns to group by when creating rolling features, by default None.

        Returns
        -------
        pl.DataFrame
            Dataframe with the created rolling features.
        """
        df = df.clone()
        windows = self.feature_config.rolling_features["windows"]
        functions = self.feature_config.rolling_features["functions"]

        if group_cols:
            for window in windows:
                for func in functions:
                    col_name: str = f"{target_col}_rolling_{window}_{func}"
                    if func == "mean":
                        df = df.with_columns(
                            pl.col(target_col).rolling_mean(window, min_samples=1).over(group_cols).alias(col_name)
                        )
                    if func == "std":
                        df = df.with_columns(
                            pl.col(target_col).rolling_std(window, min_samples=1).over(group_cols).alias(col_name)
                        )
                    if func == "std":
                        df = df.with_columns(
                            pl.col(target_col).rolling_std(window, min_samples=1).over(group_cols).alias(col_name)
                        )
                    if func == "min":
                        df = df.with_columns(
                            pl.col(target_col).rolling_min(window, min_samples=1).over(group_cols).alias(col_name)
                        )
                    if func == "max":
                        df = df.with_columns(
                            pl.col(target_col).rolling_max(window, min_samples=1).over(group_cols).alias(col_name)
                        )
        else:
            for window in windows:
                for func in functions:
                    col_name = f"{target_col}_rolling_{window}_{func}"
                    if func == "mean":
                        df = df.with_columns(pl.col(target_col).rolling_mean(window, min_samples=1).alias(col_name))
                    if func == "std":
                        df = df.with_columns(pl.col(target_col).rolling_std(window, min_samples=1).alias(col_name))
                    if func == "std":
                        df = df.with_columns(pl.col(target_col).rolling_std(window, min_samples=1).alias(col_name))
                    if func == "min":
                        df = df.with_columns(pl.col(target_col).rolling_min(window, min_samples=1).alias(col_name))
                    if func == "max":
                        df = df.with_columns(pl.col(target_col).rolling_max(window, min_samples=1).alias(col_name))
        return df

    def create_interaction_features(self, df: pl.DataFrame, categorical_cols: list[str]) -> pl.DataFrame:
        """
        Create interaction features between categorical columns.

        Creates new features by combining all pairs of categorical columns with "_interaction" suffix.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.
        categorical_cols : list[str]
            List of categorical columns to create interactions for.

        Returns
        -------
        pl.DataFrame
            DataFrame with created interaction features.
        """
        df = df.clone()

        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i + 1 :]:
                df = df.with_columns(
                    (pl.col(col1).cast(pl.Utf8) + "_" + pl.col(col2).cast(pl.Utf8)).alias(f"{col1}_{col2}_interaction")
                )

        return df

    def create_cyclical_features(self, df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """
        Create cyclical features from a date column.

        Cyclical features are created using the sine and cosine of the day of month, month of year, and day of week.
        This is useful for modeling periodic patterns in data.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.
        date_col : str, optional
            Column name of the date column. Defaults to "date".

        Returns
        -------
        pl.DataFrame
            DataFrame with created cyclical features.
        """
        df = df.clone()

        return df.with_columns(
            # month (convert 1-12 to 0-11 for proper cyclical encoding)
            pl.col(date_col).dt.month().map_elements(lambda x: np.sin(2 * np.pi * (x - 1) / 12)).alias("month_sin"),
            pl.col(date_col).dt.month().map_elements(lambda x: np.cos(2 * np.pi * (x - 1) / 12)).alias("month_cos"),
            # day (Retain original values; 1-31)
            pl.col(date_col).dt.day().map_elements(lambda x: np.sin(2 * np.pi * x / 31)).alias("day_sin"),
            pl.col(date_col).dt.day().map_elements(lambda x: np.cos(2 * np.pi * x / 31)).alias("day_cos"),
            # day of week (convert 1-7 to 0-6 for proper cyclical encoding)
            pl.col(date_col).dt.weekday().map_elements(lambda x: np.sin(2 * np.pi * (x - 1) / 7)).alias("day_of_week_sin"),
            pl.col(date_col).dt.weekday().map_elements(lambda x: np.cos(2 * np.pi * (x - 1) / 7)).alias("day_of_week_cos"),
        )

    def handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Handle missing values in a DataFrame.

        For columns with missing values, forward fill then backward fill if the column name contains "lag" or "rolling".
        Otherwise, fill with the mean of the column.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with filled missing values.
        """
        numeric_columns: list[str] = df.select(cs.numeric()).columns

        for col in numeric_columns:
            if df[col].is_null().any():
                if "lag" in col or "rolling" in col:
                    # Forward fill then backward fill
                    df = df.with_columns(pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward"))
                else:
                    # For other columns, just use average
                    df = df.with_columns(pl.col(col).fill_null(strategy="mean"))
        return df

    def create_all_features(
        self,
        df: pl.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        group_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Perform all feature engineering steps on a DataFrame.

        This includes creating date features, lag features, rolling features, cyclical features,
        interaction features, and handling missing values.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.
        target_col : str, optional
            Column name of the target column. Defaults to "sales".
        date_col : str, optional
            Column name of the date column. Defaults to "date".
        group_cols : list[str] | None, optional
            Columns to group by when creating lag and rolling features. Defaults to None.
        categorical_cols : list[str] | None, optional
            Columns to create interaction features for. Defaults to None.

        Returns
        -------
        pl.DataFrame
            DataFrame with created features.
        """
        print("Starting feature engineering pipeline")

        if group_cols:
            df = df.sort(by=group_cols + [date_col])
        else:
            df = df.sort(by=[date_col])

        # Create date features
        df = self.create_date_features(df, date_col=date_col)

        # Create lag features
        df = self.create_lag_features(df, target_col=target_col, group_cols=group_cols)

        # Create rolling features
        df = self.create_rolling_features(df, target_col=target_col, group_cols=group_cols)

        # Create cyclical features
        df = self.create_cyclical_features(df, date_col=date_col)

        # Create interaction features
        if categorical_cols:
            df = self.create_interaction_features(df, categorical_cols=categorical_cols)

        # Handle missing values
        df = self.handle_missing_values(df)

        print(f"Feature engineering pipeline completed. {len(df.columns)!r} total features.")

        return df

    def select_features(self, df: pl.DataFrame, target_col: str, importance_threshold: float = 0.001) -> list[str]:
        """
        Select features based on importance threshold.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.
        target_col : str
            Column name of the target column.
        importance_threshold : float, optional
            Minimum importance for a feature to be selected. Defaults to 0.001.

        Returns
        -------
        list[str]
            List of selected feature names.
        """
        X = df.drop(["date", target_col])
        y = df[target_col]
        cat_cols: list[str] = X.select(cs.string()).columns

        label_encoders: dict[str, LabelEncoder] = {}

        # Encode categorical variables
        for col in cat_cols:
            le = LabelEncoder()
            values = le.fit_transform(X[col])
            X = X.with_columns(pl.Series(col, values=values, dtype=pl.Int8))
            label_encoders[col] = le

        # Train the model
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)

        # Get feature importances
        importances = pl.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort(
            "importance", descending=True
        )
        # Select features based on importance
        selected_features: list[str] = importances.filter(pl.col("importance") > importance_threshold)["feature"].to_list()
        print(f"Selected features: {len(selected_features)} out of {len(X.columns)}")

        return selected_features

    def create_target_encoding(
        self,
        df: pl.DataFrame,
        target_col: str,
        categorical_cols: list[str],
        smoothing: float = 1.0,
    ) -> pl.DataFrame:
        """
        Create target encoding for categorical variables with smoothing.

        Parameters
        ----------
        df : polars.DataFrame
            Input DataFrame.
        target_col : str
            Target column name.
        categorical_cols : list[str]
            List of categorical column names.
        smoothing : float, default=1.0
            Smoothing parameter for target encoding.

        Returns
        -------
        polars.DataFrame
            DataFrame with target encoding features added.
        """
        df = df.clone()

        for col_name in categorical_cols:
            # Calculate mean target for each category
            mean_target = df.group_by(col_name).agg(pl.col(target_col).mean())
            global_mean: float = df[target_col].mean().item()  # type: ignore
            # Calculate count for each category
            count = df[col_name].value_counts()

            smooth_mean: dict = {}
            for cat in count[col_name]:
                n = count.filter(pl.col(col_name).eq(cat))["count"].item()
                smooth_mean[cat] = (
                    n * mean_target.filter(pl.col(col_name).eq(cat))[target_col].item() + smoothing * global_mean
                ) / (n + smoothing)
            df = df.with_columns(
                pl.col(col_name)
                .map_elements(lambda x: smooth_mean.get(x, global_mean))  # noqa: B023
                .alias(f"{col_name}_target_encoded")
            )
        print(f"Created target encoding for {len(categorical_cols)!r} categorical features")

        return df
