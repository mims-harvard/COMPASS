import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class NoScaler:
    def __init__(self):
        pass

    def fit(self, dfx):
        return self

    def transform(self, dfx):
        return dfx

    def fit_transform(self, dfx):
        return dfx

    def inverse_transform(self, dfx):
        return dfx


class P2Normalizer:
    def __init__(self, eps=1e-12):
        self.eps = eps

    def _square(self, x):
        return x**2

    def _clip(self, x):
        if x < self.eps:
            x = self.eps
        return x

    def fit(self, dfx):
        p2 = np.sqrt(dfx.apply(self._square, axis=1).sum())
        self.p2 = p2.apply(self._clip)
        dfx_normed = dfx / self.p2
        return dfx_normed

    def transform(self, dfx):
        df_scaled = dfx / self.p2
        return df_scaled


class Datascaler:

    def __init__(self, scale_method="minmax"):
        self.scale_method = scale_method
        if scale_method == "minmax":
            scaler = MinMaxScaler()
        elif scale_method == "standard":
            scaler = StandardScaler()
        elif scale_method == "p1norm":
            scaler = P2Normalizer()
        else:
            scaler = NoScaler()
        self.scaler = scaler

    def fit(self, dfcx):

        ## do not fit the first column
        dfx = dfcx[dfcx.columns[1:]]
        dfc = dfcx[dfcx.columns[0]]

        df = np.log2(dfx + 1)
        self.scaler.fit(df)
        return self

    def transform(self, dfcx):

        ## do not transform the first column
        dfx = dfcx[dfcx.columns[1:]]
        dfc = dfcx[[dfcx.columns[0]]]

        df = np.log2(dfx + 1)
        X = self.scaler.transform(df)
        dfx_scaled = pd.DataFrame(X, columns=dfx.columns, index=dfx.index)
        df_scaled = dfc.join(dfx_scaled)

        return df_scaled

    def fit_transform(self, dfx):
        self = self.fit(dfx)
        df_scaled = self.transform(dfx)
        return df_scaled
