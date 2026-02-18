import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List


class DataPreprocessor:
    # Data handling
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.crop_encoder = LabelEncoder()
        self.num_cols: List[str] = [
            "Temperature_C",
            "pH",
            "VFA_mgL",
            "VS_percent",
            "CN_Ratio",
            "Lignin_percent",
        ]

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Run_ID" in df.columns:
            df = df.sort_values(["Run_ID", "Day"])
            df[self.num_cols] = df.groupby("Run_ID")[self.num_cols].ffill()

        for col in self.num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._fill_missing(df)

        df["Crop"] = self.crop_encoder.fit_transform(df["Crop"])
        df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._fill_missing(df)

        df["Crop"] = self.crop_encoder.transform(df["Crop"])
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df
