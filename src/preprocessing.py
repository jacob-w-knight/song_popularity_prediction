"""
Data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SpotifyPreprocessor:
    """Preprocess Spotify music data"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = None

    def clean_data(self, df):
        """
        Remove duplicates and irrelevant features

        Parameters
        ----------
        df : pd.DataFrame
            Raw data

        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        # Remove duplicates
        df = df.drop_duplicates()

        # Drop non-explanatory features
        df = df.drop(columns=["track_id", "artists", "album_name", "track_name"])

        return df

    def encode_features(self, df):
        """
        Encode categorical variables

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned data

        Returns
        -------
        pd.DataFrame
            Encoded data
        """
        # Map explicit column to binary
        df["explicit"] = df["explicit"].map({False: 0, True: 1})

        # One-hot encode key and time signature
        df = pd.get_dummies(df, columns=["key"], prefix="key", drop_first=True)
        df = pd.get_dummies(
            df, columns=["time_signature"], prefix="meter", drop_first=True
        )

        return df

    def scale_features(self, df, numeric_cols, scaler_type="standard"):
        """
        Scale numerical features

        Parameters
        ----------
        df : pd.DataFrame
            Data with encoded features
        numeric_cols : list
            Columns to scale
        scaler_type : str, optional
            'standard' or 'minmax'

        Returns
        -------
        pd.DataFrame
            Scaled data
        """
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()

        # Scale only numeric columns
        scaled_data = self.scaler.fit_transform(df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)

        # Combine with non-numeric columns
        df_scaled = pd.concat(
            [df.drop(columns=numeric_cols).reset_index(drop=True), scaled_df], axis=1
        )

        return df_scaled
