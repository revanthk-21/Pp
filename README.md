import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_tests(series, name=None, verbose=True):
    """
    Run ADF (3 specifications) and KPSS (2 specifications) on a series
    and print a results table.

    Parameters
    ----------
    series : pd.Series or array-like
    name : str, optional — label for output (defaults to series.name)
    verbose : bool, default True — print formatted table

    Returns
    -------
    pd.DataFrame with columns: Variable, Test Name, Test Statistic, p-value, Conclusion
    """
    s = pd.Series(series).dropna()
    if name is None:
        name = s.name if s.name is not None else "series"

    rows = []

    adf_specs = [
        ("ADF: No deterministic terms", "n"),
        ("ADF: Constant", "c"),
        ("ADF: Constant and linear trend", "ct"),
    ]
    for label, reg in adf_specs:
        try:
            stat, pval, *_ = adfuller(s, regression=reg, autolag="AIC")
            conclusion = "Stationary" if pval < 0.05 else "Non-Stationary"
        except Exception as e:
            stat, pval, conclusion = np.nan, np.nan, f"Error: {e}"
        rows.append([name, label, stat, pval, conclusion])

    kpss_specs = [
        ("KPSS: Constant", "c"),
        ("KPSS: Constant and linear trend", "ct"),
    ]
    for label, reg in kpss_specs:
        try:
            stat, pval, *_ = kpss(s, regression=reg, nlags="auto")
            # KPSS null = stationary -> low p-value means NON-stationary
            conclusion = "Stationary" if pval >= 0.05 else "Non-Stationary"
        except Exception as e:
            stat, pval, conclusion = np.nan, np.nan, f"Error: {e}"
        rows.append([name, label, stat, pval, conclusion])

    results = pd.DataFrame(
        rows, columns=["Variable", "Test Name", "Test Statistic", "p-value", "Conclusion"]
    )

    if verbose:
        print(f"\nStationarity Tests: {name}\n")
        display_df = results.copy()
        display_df["Test Statistic"] = display_df["Test Statistic"].map(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "NA"
        )
        display_df["p-value"] = display_df["p-value"].map(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "NA"
        )
        try:
            from IPython.display import display
            display(display_df)
        except Exception:
            print(display_df.to_string(index=False))

    return results
# raw DV
stationarity_tests(df['deriv_notl_co_industry_dln'])

# on your GFC-excluded / deseasonalized residual
stationarity_tests(resid, name='deriv_notl_co_industry_dln (deseasonalized, ex-GFC)')
