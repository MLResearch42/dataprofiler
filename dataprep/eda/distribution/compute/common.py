"""Common used methods for plot function"""

from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from ...dtypes import LatLong
from ...eda_frame import to_pandas


def _gen_latlong(df: pd.DataFrame, x: LatLong) -> pd.Series:
    """
    Merge the latitude and longitude columns into one new column of tuples.
    Rows where either coordinate is missing become missing.
    """

    # Make sure the new column name is not contained
    columns = df.columns
    name = x.lat + "_" + x.long
    i = 0
    while name in columns:
        name = f"{name}_{i}"
        i += 1

    lat, long = df[x.lat], df[x.long]
    missing = (lat.isna() | long.isna()).to_numpy()
    pairs = [None if miss else pair for pair, miss in zip(zip(lat, long), missing)]
    return pd.Series(pairs, name=name, index=df.index, dtype=object)


def gen_new_df_with_used_cols(
    df: Any,
    x: Optional[Union[str, LatLong]] = None,
    y: Optional[Union[str, LatLong]] = None,
    z: Optional[str] = None,
) -> Tuple[Dict[Optional[Union[str, LatLong]], Optional[str]], pd.DataFrame]:
    """
    Keep only used columns in x, y, z, and gen new Latlong columns if any x, y, z is LatLong.
    """

    # The key of new_names is the old name, and value is
    # new name for x, y, z. For non-LatLong type, it's the original name,
    # Otherwise it's the generated column name.
    new_names: Dict[Optional[Union[str, LatLong]], Optional[str]] = {}

    pd_df = to_pandas(df)
    if x is None and y is None and z is None:
        return {}, pd_df

    # Column labels may be non-strings in the user's frame while x, y, z are given
    # as strings: look columns up by their string form.
    by_str = {}
    for pos, label in enumerate(pd_df.columns):
        by_str.setdefault(str(label), pos)

    def pick(name: str) -> pd.Series:
        if name not in by_str:
            raise KeyError(
                f"Column {name!r} not found. Available columns: {list(by_str)[:20]}"
                + (" ..." if len(by_str) > 20 else "")
            )
        srs = pd_df.iloc[:, by_str[name]]
        return srs.rename(name)

    # new_srss is the used columns. It's dict because of avoid duplicate names
    new_srss: Dict[str, pd.Series] = {}
    for col in (x, y, z):
        if isinstance(col, LatLong):
            both = pd.concat([pick(col.lat), pick(col.long)], axis="columns")
            lat_long = _gen_latlong(both, col)
            new_names[col] = lat_long.name
            new_srss[lat_long.name] = lat_long
            new_srss[col.lat] = both[col.lat]
            new_srss[col.long] = both[col.long]
        else:
            new_names[col] = None if col is None else str(col)
            if col is not None:
                new_srss[str(col)] = pick(str(col))

    new_df = pd.concat(list(new_srss.values()), axis="columns")

    return new_names, new_df
