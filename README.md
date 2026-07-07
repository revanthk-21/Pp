import pandas as pd
import numpy as np
import statsmodels.api as sm

def get_clean_residual(df, date_col, dv_col, gfc_start='2008-09-01', gfc_end='2009-06-30'):
    """
    Regresses out (1) a GFC shock dummy and (2) quarterly seasonal dummies
    from the DV, returning the residual for stationarity testing.
    """
    data = df[[date_col, dv_col]].copy().dropna()
    data[date_col] = pd.to_datetime(data[date_col])

    # 1. GFC dummy
    data['gfc_dummy'] = ((data[date_col] >= gfc_start) & (data[date_col] <= gfc_end)).astype(int)

    # 2. Quarter dummies (Q1 as baseline, so include Q2/Q3/Q4)
    data['quarter'] = data[date_col].dt.quarter
    quarter_dummies = pd.get_dummies(data['quarter'], prefix='Q', drop_first=True).astype(int)

    X = pd.concat([data['gfc_dummy'], quarter_dummies], axis=1)
    X = sm.add_constant(X)
    y = data[dv_col]

    model = sm.OLS(y, X, missing='drop').fit()
    print(model.summary())

    residual = pd.Series(model.resid, index=data.index, name=f"{dv_col}_resid")
    return residual, model

# Usage:
resid, model = get_clean_residual(df, date_col='date', dv_col='deriv_notl_co_industry_dln')

# then test it:
stationarity_tests(resid, name='deriv_notl_co_industry_dln (GFC + seasonal removed)')
