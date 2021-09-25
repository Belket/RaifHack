import pandas as pd
from raif_hack.utils import UNKNOWN_VALUE
from raif_hack.nan_fighter import NanFighter


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()

    df_new = NanFighter.fight(df_new)

    fillna_cols = ['region', 'city', 'street', 'realty_type']
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new
