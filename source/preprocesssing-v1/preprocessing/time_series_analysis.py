import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL
from utils import csv
from preprocessing.visualization import finish


def analyze(train, out):
    """ACF/PACF gợi ý p,q; STL mô tả trend/seasonal/residual, không làm feature."""
    price = train['Gold Price'].dropna()
    for name, series in [('price', price), ('logret', np.log(price).diff().dropna())]:
        lags = min(50, len(series) // 2 - 1)
        a, ac = acf(series, nlags=lags, alpha=.05)
        p, pc = pacf(series, nlags=lags, alpha=.05)
        table = pd.DataFrame({
            'lag': range(lags + 1),
            'acf': a, 'pacf': p,
            'acf_lower': ac[:,0], 'acf_upper': ac[:,1],
            'pacf_lower': pc[:,0], 'pacf_upper': pc[:,1]
        })
        csv(table, out / f'reports/{name}_acf_pacf.csv')
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        for ax, key in zip(axes, ['acf', 'pacf']):
            ax.stem(table['lag'], table[key])
            ax.fill_between(table['lag'].to_numpy(), (table[key+'_lower']-table[key]).to_numpy(), (table[key+'_upper']-table[key]).to_numpy(), alpha=.2)
            ax.set_title(f'{name}: {key.upper()}')
        finish(fig, out / f'figures/{name}_acf_pacf.png')
    result = STL(price, period=252, robust=True).fit()
    csv(
        pd.DataFrame({'Date': train.loc[price.index, 'Date'], 'trend': result.trend, 'seasonal': result.seasonal, 'residual': result.resid}),
        out / 'reports/stl_train.csv'
    )
    finish(result.plot(), out / 'figures/stl_train.png')

    # Độ mạnh = max(0, 1 - Var(R)/Var(component+R)).
    rows = []
    for name, component in [('trend', result.trend), ('seasonality', result.seasonal)]:
        denominator = np.var(component + result.resid)
        strength = max(0, 1 - np.var(result.resid)/denominator) if denominator > 0 else np.nan
        rows.append(dict(component=name, period=252, strength=strength))
    csv(pd.DataFrame(rows), out/'reports/stl_strength.csv')
