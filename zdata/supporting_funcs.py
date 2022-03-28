from edss_fetch import EDSsession
from typing import List, Dict, Text, Type, Tuple
from copy import deepcopy
import pandas as pd


def flatten_dict(dct: dict) -> dict:
    d = deepcopy(dct)
    for key, value in d.items():
        if isinstance(value, dict):
            del d[key]
            return {**d, **flatten_dict(value)}
    return d


def extract_data(dat: List[dict], key: str = 'data', col_iter: int = 0) -> Tuple[dict, dict]:
    # time series data
    data = flatten_dict(dat[col_iter])
    ts_data = data[key]

    # meta data
    meta_data = deepcopy(data)
    del meta_data[key]

    return {ts_data[i][0]: ts_data[i][1] for i, _ in enumerate(ts_data)}, meta_data


def create_dfs(ts: dict, meta: dict) -> Tuple[dict, dict]:
    """
    return two dfs: prices and meta data
    """
    meta_df = pd.DataFrame([meta])
    for dt_col in ['updated_at', 'created_at']:
        if meta.get(dt_col) is not None:
            meta_df[dt_col] = pd.to_datetime(meta_df[dt_col]).dt.ceil(freq='s')

    ts_df = pd.DataFrame(ts.items(), columns=['date', 'value'])
    ts_df['date'] = pd.to_datetime(ts_df['date']).dt.date
    ts_df.set_index('date', inplace=True)

    # sort, change index type and to end of month
    ts_df = ts_df.sort_index()
    ts_df.index = ts_df.index.astype('datetime64[ns]')
    ts_df = ts_df.to_period('M').to_timestamp('M')

    for ts_name in ['ticker', 'code']:
        if meta.get(ts_name) is not None:
            ts_df.rename(columns={'value': meta_df[ts_name].values[0]}, inplace=True)

    return ts_df, meta_df


def create_agg_dfs(dat: List[dict], key: str = 'data') -> Tuple[dict, dict]:
    """
    return two dfs: prices and meta data
    """
    all_ts = pd.DataFrame()
    all_meta = pd.DataFrame()

    for n, _ in enumerate(dat):
        ts, meta = create_dfs(*extract_data(dat=dat, key=key, col_iter=n))
        all_ts = pd.concat([all_ts, ts], axis=1)
        all_meta = pd.concat([all_meta, meta], axis=0)

    all_meta.reset_index(drop=True, inplace=True)
    return all_ts, all_meta


def fetch_bulk_data(session: Type[EDSsession], codes: List[str], kind: str = 'benchmark') -> List:
    if kind not in ('benchmark', 'fund'):
        raise ValueError(f"kind must be either benchmark or fund; received {kind}")

    get_func = {
        'fund': session.get_fund,
        'benchmark': session.get_benchmark
    }
    bulk_data = []
    for code in codes:
        code_data = get_func[kind](code)
        if not code_data:
            print(f'no data: {code}')
        else:
            bulk_data.append(code_data[0])

    return bulk_data
