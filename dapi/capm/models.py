import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from collections.abc import Iterable
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import List, Dict, Text, Type, Tuple
# from copy import deepcopy
from edss_fetch import EDSsession
from supporting_funcs import flatten_dict, extract_data, create_dfs, create_agg_dfs


eds_email = 'info@economicdatasciences.com'
eds_pass = 'Xp6UywFJ8fG!H6th'


class AllFactors:
    FACTORS_TS = None
    FACTORS_META = None

    def __init__(self, session: Type[EDSsession] = None):
        self.update_factors(session=session)

    @classmethod
    def update_factors(cls, session: Type[EDSsession] = None):
        if session is None:
            session = EDSsession(eds_email=eds_email, eds_pass=eds_pass)

        all_factors = session.get_factors()
        cls.FACTORS_TS, cls.FACTORS_META = create_agg_dfs(all_factors, key='values')


class Factor(AllFactors):
    # risk_free_rate -> should specify it must take a Factor
    def __init__(self, code: str, risk_free_rate=None, session: Type[EDSsession] = None):
        if self.FACTORS_TS is None or self.FACTORS_META is None:
            super().__init__(session=session)

        self.code = code
        self.unadjusted_returns, self.meta = self.get_factor(code=code)

        if risk_free_rate is not None:
            self.returns = self.unadjusted_returns - risk_free_rate.returns
        else:
            self.returns = self.unadjusted_returns

        self.name = self.meta['name'].values[0]
        self.description = self.meta['description'].values[0]
        self.region = self.meta['region'].values[0]
        self.factor_type = self.meta['factor_type'].values[0]

    def get_factor(self, code: str):
        return self.FACTORS_TS[code], self.FACTORS_META[self.FACTORS_META.code == code]


class BenchmarkAssetsByClass:
    BABC_TS = {}
    BABC_META = {}

    def __init__(self, asset_class: str, session: Type[EDSsession] = None):
        self.update_babc(asset_class=asset_class, session=session)

    @classmethod
    def update_babc(cls, asset_class: str, session: Type[EDSsession] = None):
        if session is None:
            session = EDSsession(eds_email=eds_email, eds_pass=eds_pass)

        assets = session.get_benchmark_assets_by_class(asset_class=asset_class)
        cls.BABC_TS[asset_class], cls.BABC_META[asset_class] = create_agg_dfs(assets)


class Asset(BenchmarkAssetsByClass):
    def __init__(self,
                 asset_class: str,
                 ticker: str,
                 risk_free_rate: Type[Factor] = None,
                 session: Type[EDSsession] = None):
        if self.BABC_TS.get(asset_class) is None or self.BABC_META.get(asset_class) is None:
            super().__init__(asset_class=asset_class, session=session)

        self.prices, self.meta = self.get_asset(asset_class=asset_class, ticker=ticker)
        self.prices = self.prices.sort_index()
        self.unadjusted_returns = self.prices.pct_change()

        if risk_free_rate is not None:
            self.returns = self.unadjusted_returns - risk_free_rate.returns
        else:
            self.returns = self.unadjusted_returns

        self.asset_class = asset_class
        self.benchmark_code = self.meta['benchmark_code'].values[0]
        self.currency_code = self.meta['currency_code'].values[0]
        self.ticker = ticker
        self.tr_code = self.meta['tr_code'].values[0]
        self.type_code = self.meta['type_code'].values[0]
        self.name = self.meta['name'].values[0]
        self.code = self.meta['code'].values[0]

    def get_asset(self, asset_class: str, ticker: str):
        return (self.BABC_TS[asset_class][ticker],
                self.BABC_META[asset_class].loc[self.BABC_META[asset_class].ticker == ticker])


class Fund:
    def __init__(self, code: str, risk_free_rate: Type[Factor] = None, session: Type[EDSsession] = None):
        if session is None:
            session = EDSsession(eds_email=eds_email, eds_pass=eds_pass)

        self.levels, self.meta = create_dfs(*extract_data(session.get_fund(code=code)))
        self.code = code
        self.unadjusted_returns = self.levels.pct_change()
        self.unadjusted_returns = self.returns.iloc[:, 0]  # convert single column df to series

        if risk_free_rate is not None:
            self.returns = self.unadjusted_returns - risk_free_rate.returns
        else:
            self.returns = self.unadjusted_returns

        self.name = self.meta['name'].values[0]
        self.frequency = self.meta['frequency'].values[0]
        self.currency = self.meta['currency'].values[0]
        self.yld = self.meta['yld'].values[0]


class Benchmark:
    def __init__(self, code: str, risk_free_rate: Type[Factor] = None, session: Type[EDSsession] = None):
        if session is None:
            session = EDSsession(eds_email=eds_email, eds_pass=eds_pass)

        self.ts, self.meta = create_dfs(*extract_data(session.get_benchmark(code=code)))
        self.code = code
        if self.meta['yld'].values[0]:
            self.unadjusted_returns = self.ts.iloc[:, 0]
            self.levels = None
        else:
            self.levels = self.ts.iloc[:, 0]
            self.unadjusted_returns = self.levels.pct_change()

        if risk_free_rate is not None:
            self.returns = self.unadjusted_returns - risk_free_rate.returns
        else:
            self.returns = self.unadjusted_returns

        self.name = self.meta['name'].values[0]
        self.frequency = self.meta['frequency'].values[0]
        self.currency = self.meta['currency'].values[0]
        self.yld = self.meta['yld'].values[0]


class FactorCollection:
    def __init__(self):
        self.factors = []

    def add(self, factor):
        if isinstance(factor, Factor) and factor not in self.factors:
            self.factors.append(factor)


class ExpectedType:
    """
    Checks that the types of values passed in to a class at initiation are as expected
    """

    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if value is not None:
            if isinstance(value, Iterable):
                if isinstance(self.expected_type, Iterable):
                    for elem in value:
                        if elem in self.expected_type:
                            raise TypeError('not all elements are as expected ' + str(self.expected_type))
                else:
                    raise TypeError('expected single value, received iterable')
            elif isinstance(self.expected_type, Iterable):
                if not isinstance(value, tuple(self.expected_type)):
                    raise TypeError('expected ' + str(self.expected_type))
            else:
                if not isinstance(value, self.expected_type):
                    raise TypeError('expected ' + str(self.expected_type))
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


def type_assert(**kwargs):
    def decorate(cls):
        for name, expected_type in kwargs.items():
            setattr(cls, name, ExpectedType(name, expected_type))
        return cls

    return decorate


# @type_assert(asset=Stock, risk_free_rate=RiskFree, factors=[Factor, FactorCollection])
class PredictionModel:
    """
    variables:
        - asset : a Stock object
        - risk_free_rates : a RiskFree object
        - factors : either a Factor object or a collection of Factor objects
    """

    def __init__(self):
        self.asset = None
        self.factors = None
        self.fitted_df = None

    def _is_setup_complete(self):
        if self.asset is not None and self.factors is not None:
            return True
        else:
            return False

    def add(self, obj, *, override=False):
        if isinstance(obj, Asset) or isinstance(obj, Fund):
            if self.asset is None or override is True:
                self.asset = obj
        if isinstance(obj, (Factor, FactorCollection)):
            # if a FactoryCollection has already been added, it can only be overridden
            if isinstance(self.factors, FactorCollection):
                if override:
                    self.factors = obj
                else:
                    raise ValueError('set override=True to replace the current FactoryCollection')

            if override:
                self.factors = obj
            elif isinstance(self.factors, list):
                if obj not in self.factors:
                    # TODO: maybe check if the object asset_ids/names are the same?
                    self.factors.append(obj)
                else:
                    raise ValueError(f'same object {obj} has already been added')
            elif self.factors is None:
                if isinstance(obj, Factor):
                    self.factors = [obj]
                elif isinstance(obj, FactorCollection):
                    self.factors = obj
            else:
                raise ValueError(f'set override=True to replace the current Factor')

    def fit(self):
        if self._is_setup_complete():
            fitted_df = pd.DataFrame()
            fitted_df = fitted_df.append(pd.Series(self.asset.returns, name=self.asset.name))

            if isinstance(self.factors, Factor):
                expl_vars = [pd.Series(self.factors.returns, name=self.factors.name)]
            elif isinstance(self.factors, list):
                expl_vars = [pd.Series(fct.returns, name=fct.name) for fct in self.factors]
            elif isinstance(self.factors, FactorCollection):
                expl_vars = [pd.Series(fct.returns, name=fct.name) for fct in self.factors.factors]
            else:
                raise TypeError(self.factors)

            fitted_df = fitted_df.append(expl_vars)
            # TODO: provide statistics around dropped records
            fitted_df = fitted_df.T.dropna()
            self.fitted_df = fitted_df

    def predict(self, method='recursive', *, shift_window=36, alpha=0.05, hl=None):
        if self.fitted_df is not None:
            # number of observations
            m_dim = self.fitted_df.shape[0]

            # the asset's excess return (explained variable)
            y = self.fitted_df.iloc[:, 0]

            # explanatory variable(s)
            X = self.fitted_df.iloc[:, 1:]

            if shift_window >= m_dim:
                raise ValueError(f"not enough records {m_dim} to accommodate shift_window {shift_window}")

            agg_summary = np.array([[]] * 8)

            for m in range(shift_window, m_dim):
                # if half-life is not specified, set weights to 1 => same as OLS
                if method == 'recursive':
                    weights = np.exp(hl * np.linspace(1, m, m - 1))[::-1] if hl is not None else 1
                    wls_model = sm.WLS(y[:m - 1], X.iloc[:m - 1, :], weights=weights).fit()
                elif method == 'rolling':
                    weights = np.exp(hl * np.linspace(1, shift_window, shift_window - 1)) if hl is not None else 1
                    wls_model = sm.WLS(y[m - shift_window:m - 1], X.iloc[m - shift_window:m - 1, :],
                                       weights=weights).fit()
                else:
                    raise ValueError(f"{method} not accepted; try: recursive or rolling")

                # standard error of regression (ssr / (n-p))
                se = np.sqrt(wls_model.scale)

                # prediction for the following period (one period)
                prediction = wls_model.predict(X.iloc[m].values)

                prediction_summary = [
                    X.iloc[m].name,  # prediction date
                    wls_model.params.values,  # beta estimate(s)
                    prediction,  # prediction value
                    se,  # std error of regression
                    prediction - stats.norm.ppf(1 - 0.05 / 2) * se,  # lower pred intvl
                    prediction + stats.norm.ppf(1 - 0.05 / 2) * se,  # higher pred intvl
                    y[m],  # actual excess return
                    (y[m] - prediction) / se  # pred err (no of std errs)
                ]

                prediction_summary = np.array(prediction_summary, dtype='object')

                # aggregate prediction summaries
                agg_summary = np.c_[agg_summary, prediction_summary]

            # aggregate predictions
            col_names = ['date', 'param_estimate', 'prediction', 'stderr', 'lower_pi', 'upper_pi', 'actual', 'zscore']
            df_agg_summary = pd.DataFrame(agg_summary).T
            df_agg_summary.columns = col_names

            # when the observed (actual) value falls outside the prediction interval -> outlier
            df_agg_summary['outlier'] = df_agg_summary.apply(
                lambda row: False if row.lower_pi <= row.actual <= row.upper_pi else True, axis=1
            )

            # update index to reflect the shift/window length
            index_update_dict = {
                'recursive': df_agg_summary.set_index(np.arange(shift_window, m_dim)),
                'rolling': df_agg_summary.set_index(X.index[shift_window:])
            }

            df_agg_summary = index_update_dict[method]

            # add attributes so that they can be used for plotting
            df_agg_summary.attrs['shift_window'] = shift_window
            df_agg_summary.attrs['method'] = method
            df_agg_summary.attrs['hl'] = hl
            df_agg_summary.attrs['alpha'] = alpha
            return df_agg_summary

    def fit_predict(self, method='recursive', *, shift_window=36, alpha=0.05, hl=None):
        self.fit()
        return self.predict(method=method, shift_window=shift_window, alpha=alpha, hl=hl)

    def info(self):
        if self._is_setup_complete():
            print('Model setup')
            print('-' * 50)

            print(f'Stock: {self.asset.asset_id} ({self.asset.name})')
            print(f'Risk-free rate: {self.risk_free_rate.asset_id} ({self.risk_free_rate.name})')

            if isinstance(self.factors, FactorCollection):
                for i, fct in enumerate(self.factors.factors):
                    print(f'x{i + 1}: {fct.asset_id} ({fct.name})')
            if isinstance(self.factors, list):
                for i, fct in enumerate(self.factors):
                    print(f'x{i + 1}: {fct.asset_id} ({fct.name})')
        else:
            print('Model setup is incomplete')
            for component_name, component in zip(['Stock', 'Risk-free rate', 'Factor(s)'],
                                                 [self.asset, self.risk_free_rate, self.factors]):
                print(f"{component_name}: {'OK' if component is not None else 'N/A'}")

        if self.fitted_df is not None:
            print('-' * 50)
            print(f'Number of records:{len(self.fitted_df)}')
            print(f"Dates: {self.fitted_df.index.min().strftime('%b %d, %Y')} - "
                  f"{self.fitted_df.index.max().strftime('%b %d, %Y')}")
        else:
            print('Model has not been fitted yet')


def plot_predictions(results, *, pi_smoothing=None, verbose=False):
    # smoothing the edges of the prediction interval
    if pi_smoothing:
        # create a copy to avoid making changes to the dataframe passed in
        results = results.copy()
        results['upper_pi'] = results['upper_pi'].rolling(pi_smoothing).mean()
        results['lower_pi'] = results['lower_pi'].rolling(pi_smoothing).mean()
        results = results.dropna()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(results['date'].values, results['actual'].values, label='Realized excess return')

    sns.scatterplot(data=results[results.outlier],
                    x='date',
                    y='actual',
                    size=np.abs(results["zscore"].apply(lambda x: x[0])),   # temp solution
                    sizes=(10, 200),
                    color='darkred',
                    alpha=0.8,
                    label='Realized outliers',
                    legend=None
                    )

    ax.fill_between(x=results['date'].values,
                    y1=results['upper_pi'].values,
                    y2=results['lower_pi'].values,
                    color='gray',
                    alpha=0.3,
                    label=f"Prediction interval {'- smoothed ' if pi_smoothing else ''}"
                          f"({100 - results.attrs['alpha'] * 100}%)"
                    )

    ax.set_title(f"Realized excess return (prediction method:{results.attrs['method']}, "
                 f"{'shift' if results.attrs['method'] == 'recursive' else 'window size'}"
                 f"={results.attrs['shift_window']}"
                 f"{', half-life: ' + str(results.attrs['hl']) + ')' if results.attrs['hl'] else ')'}")
    ax.set_ylabel('Realized excess return')
    ax.set_xlabel('Date')
    plt.legend()

    if verbose:
        print(f'Number of outliers: {results.outlier.sum()} out of {len(results)} ' +
              f'({results.outlier.sum() / len(results):.2%})')


def factor_names(model):
    if isinstance(model.factors, FactorCollection):
        return_list = [factor.name for factor in model.factors.factors]
    else:
        return_list = [factor.name for factor in model.factors]

    if len(return_list) == 1:
        return return_list
    else:
        return [return_list]


def beta_stds(predictions):
    no_params = len(predictions.param_estimate.iloc[0])

    if no_params == 1:
        return [np.std(predictions.param_estimate)]
    else:
        params = pd.DataFrame(predictions.param_estimate.values.tolist())
        params = np.std(params, axis=0)
        return [list(params)]


def bulk_results(model=None,
                 stocks=None,
                 risk_free_rates=None,
                 factors=None,
                 method='recursive',
                 shift_window_start=12,
                 shift_window_end=60,
                 step=12,
                 alpha=0.05,
                 hl_applied=False,
                 hl_start=-0.01,
                 hl_end=0,
                 hl_step=20):
    models = []
    if model is not None and not isinstance(model, PredictionModel):
        raise TypeError(f"{type(model)}, expected: None or {type(PredictionModel)}")
    elif model is not None:
        models.append(model)

    # this block is skipped if exact model is provided through the model parameter
    if not models:
        components = [stocks, risk_free_rates, factors]
        for i, component in enumerate(components):
            if component is None:
                raise ValueError('either model or [stocks, risk_free_rates, factors] all have to be provided')
            if not isinstance(component, Iterable):
                components[i] = [components[i]]

        model_variations = product(*[range(len(component)) for component in components], repeat=1)
        for i, variation in enumerate(model_variations):
            pred_model = PredictionModel()
            for idx, component in zip(variation, components):
                pred_model.add(component[idx])
            models.append(pred_model)

    if hl_applied:
        hls = np.linspace(hl_start, hl_end, hl_step)

        # adding 0 (OLS) if it is not included in the interval specified
        if not np.isin(0, hls):
            hls = np.insert(hls, 0, 0)
            hls.sort()
    else:
        hls = [None]

    bulk_results_df = pd.DataFrame(columns=['stock',
                                            'risk-free rate',
                                            'factors',
                                            'method',
                                            'shift_window',
                                            'half-life',
                                            'number of predictions',
                                            'number of outliers',
                                            'outliers / predictions',
                                            'stddiv of beta estimate',
                                            'stddiv of predictions',
                                            'avg width of PI',
                                            'avg pred error',
                                            'avg pred error when outlier',
                                            ])

    for sw in np.arange(shift_window_start, shift_window_end + 1, step):
        for hl in hls:
            for model_variation in models:
                model_variation.fit()
                # shift_window size must be smaller than the number of records
                if sw < len(model_variation.fitted_df):
                    predictions = model_variation.predict(method=method, shift_window=sw, alpha=alpha, hl=hl)
                    result = pd.DataFrame.from_dict({
                        'stock': [model_variation.asset.name],
                        'risk-free rate': [model_variation.risk_free_rate.name],
                        'factors': factor_names(model_variation),
                        'method': [method],
                        'shift_window': [sw],
                        'half-life': [hl],
                        'number of predictions': [len(predictions)],
                        'number of outliers': [predictions.outlier.sum()],
                        'outliers / predictions': [predictions.outlier.sum() / len(predictions)],
                        'stddiv of beta estimate': beta_stds(predictions),
                        'stddiv of predictions': [np.std(predictions.prediction)],
                        'avg width of PI': np.mean(predictions.upper_pi - predictions.lower_pi),
                        'avg pred error': np.mean(abs(predictions.zscore)),
                        'avg pred error when outlier': np.mean(abs(predictions[predictions.outlier].zscore)),
                    })

                    bulk_results_df = bulk_results_df.append(result)

    return bulk_results_df.reset_index(drop=True)
