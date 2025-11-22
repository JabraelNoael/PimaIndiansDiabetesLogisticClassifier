import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import spearmanr

def softmax(x):
    match type(x):
        case pd.DataFrame:
            output = np.zeros(x.shape)
            for i in range(x.shape[1]):
                output[:,i] = x.iloc[:,i]/x.iloc[:,i].sum()
            return pd.DataFrame(output,columns=x.columns)
        case list:
            #output = np.zeros(len(x))
            exp = np.exp(x)
            for i in range(len(x)):
                output = exp/np.sum(exp)
            return output

def wald_significance_multivariate(x_col: pd.DataFrame, y_col, alpha: float = 0.05):
   
    X = sm.add_constant(x_col, has_constant="add")
    model = sm.GLM(y_col, X, family=sm.families.Binomial(),missing ="drop")
    res = model.fit()

    pvals = res.pvalues.drop(labels=["const"], errors="ignore")
    significant_predictors = [name for name, p in pvals.items() if p < alpha]

    wald_table = pd.DataFrame({
        "coef":   res.params.drop(labels=["const"], errors="ignore"),
        "se":     res.bse.drop(labels=["const"], errors="ignore"),
        "z":      res.tvalues.drop(labels=["const"], errors="ignore"),
        "pvalue": pvals
    }).sort_values("pvalue")

    return significant_predictors, wald_table, res