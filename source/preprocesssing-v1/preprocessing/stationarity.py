import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from config import MARKET, MACRO
from utils import csv


def test_series(series, name, transform):
    """ADF H0: unit root; KPSS H0: dừng quanh mức trung bình.
    Hai kiểm định bất đồng thì kết luận chưa rõ, không tự suy ra trend-stationary.
    """
    series = series.dropna()
    result = {'series': name, 'transform': transform, 'n': len(series)}
    if len(series) < 30 or series.nunique() < 2:
        return dict(result, conclusion='insufficient_or_constant')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            adf = adfuller(series, autolag='AIC')
            k = kpss(series, regression='c', nlags='auto')
        a_reject = adf[1] < .05
        k_reject = k[1] < .05
        if a_reject and not k_reject:
            conclusion = 'evidence_stationary'
        elif not a_reject and k_reject:
            conclusion = 'evidence_nonstationary'
        else:
            conclusion = 'inconclusive'
        return dict(
            result,
            adf_lags=adf[2],
            adf_nobs=adf[3],
            **{f"adf_critical_{key}": value for key,value in adf[4].items()},
            kpss_lags=k[2],
            **{f"kpss_critical_{key}": value for key,value in k[3].items()},
            adf_stat=adf[0],
            adf_p=adf[1],
            kpss_stat=k[0],
            kpss_p=k[1],
            conclusion=conclusion,
            warnings=' | '.join(str(x.message) for x in caught)
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return dict(result, conclusion='test_failed', warnings=str(error))


def scan(train, out):
    rows = []
    for column in MARKET + MACRO:
        series = train[column]
        rows.append(test_series(series, column, 'level'))
        if column == 'Oil' or column in MACRO:
            transformed = series.diff()
            method = 'difference'
        else:
            transformed = np.log(series.where(series > 0)).diff()
            method = 'log_return'
        rows.append(test_series(transformed, column, method))
    csv(pd.DataFrame(rows), out / 'reports/stationarity.csv')
