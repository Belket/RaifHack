from copy import deepcopy
import numpy as np
import statsmodels.api as sm
from settings import EXPERT_EXTERNAL_FEATURES, SELECTORS_PIPE


class FeatureSelector:

    @staticmethod
    def expert_review(df):
        appropriate_features = list(set(df.columns) - set(EXPERT_EXTERNAL_FEATURES))
        df = df[appropriate_features]
        return df

    @staticmethod
    def stepwize_regression(df):
        df_temp = deepcopy(df)
        p_value_threshold = 0.05  # if greater, than should be dropped
        appropriate_features = []

        target = 'per_square_meter_price'

        categorial_features = [
            'city',
            'floor',
            'id',
            'osm_city_nearest_name',
            'region',
            'street',
            'realty_type',
            'price_type'
        ]

        drop_columns = categorial_features + ['date', 'per_square_meter_price']

        x_columns = list(set(df.columns) - set(drop_columns))

        current_columns = []
        for column in x_columns:
            current_columns.append(column)
            X = df_temp[current_columns].astype(float)
            y = list(df_temp[target])
            results = sm.OLS(y, X).fit()
            appropriate_features = list((results.pvalues.iloc[np.where(results.pvalues <= p_value_threshold)]).index)
        return appropriate_features

    @staticmethod
    def select_features(df):
        selectors_pipe = [getattr(FeatureSelector, section) for section in SELECTORS_PIPE]
        for section in selectors_pipe:
            df = section(df)
        return df
