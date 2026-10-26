"""
Preprocessing of stock characteristics.
"""

import pandas as pd


def rank_characteristics(
    df: pd.DataFrame,
    id_cols: list[str] = ["date", "cusip", "permno", "r_1"],
) -> pd.DataFrame:
    """
    Cross-sectional percentile ranks of characteristics, centred on zero.

    At each date, every column not in ``id_cols`` is replaced by its
    percentile rank minus 0.5 (values in ``(-0.5, 0.5]``, ties averaged).
    Missing values are then set to 0, i.e. to the cross-sectional median.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with a ``"date"`` column or index level.
    id_cols : list of str, default=["date", "cusip", "permno", "r_1"]
        Columns left unchanged (identifiers, returns).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the characteristics ranked.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_product([pd.date_range('2020-01-01', periods=100), range(50)],
    ...                                  names=['date', 'id'])
    >>> df = pd.DataFrame(np.random.randn(5000, 3), index=idx, columns=['r_1', 'mom', 'value'])
    >>> ranked = rank_characteristics(df, id_cols=['r_1'])
    """
    char_cols = [c for c in df.columns if c not in id_cols]

    def _rank_and_scale(group: pd.DataFrame) -> pd.DataFrame:
        # Percentile ranks in (0, 1], shifted to (-0.5, 0.5]; NaN -> 0
        ranked = group[char_cols].rank(pct=True, na_option="keep") - 0.5
        return ranked.fillna(0.0)

    df = df.copy()
    df[char_cols] = df.groupby("date", group_keys=False).apply(_rank_and_scale)
    return df
