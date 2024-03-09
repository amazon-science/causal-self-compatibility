# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Iterable, List, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy._typing import NDArray
from scipy.stats import chi2


class ResidualTriplet(NamedTuple):
    tau: float
    res_y: float
    res_x: float


class CovariaceResult(NamedTuple):
    est: NDArray
    covmat: NDArray


class TestResult(NamedTuple):
    pval: float
    statistic: float
    rank: int


class RoCA:
    def __init__(self, contrast: NDArray = None, rank_est: bool = False):
        """
        Perform Ro bustness test of total (causal) effect estimates via Covariate Adjustment

        Args:
        - data: data for testing
        - x: column number of exposure variable
        - y: column number of outcome variable
        - g: candidate causal DAG, an object of "matrix", "graphNEL" or "igraph"
        - strategy: one of "all" or "min+"
        - contrast: contrast matrix of appropriate dimensions
                    if None, the default contrast matrix (see paper) will be used

        """
        self.contrast = contrast
        self.rank_est = rank_est

    @staticmethod
    def get_format_strings(x: int, y: int, separtion_sets: Iterable[Iterable[int]], var_names: List[str]) -> List[str]:
        fml = []
        for s in separtion_sets:
            format_string = f"{var_names[y]} ~ {var_names[x]}"
            for node in s:
                format_string += f" + {var_names[node]}"
            fml.append(format_string)
        return fml

    @staticmethod
    def get_residuals(data: pd.DataFrame, format_string: str) -> ResidualTriplet:
        if "p44/42" in format_string:
            data["pff_ft"] = data["p44/42"]
            format_string = format_string.replace("p44/42", "pff_ft")
        fit1 = smf.ols(format_string, data=data).fit()
        split_string = format_string.split(' ~ ')
        # If the regressand is in the list of regressors, the fit is in the anti-causal direction.
        # Then we 'set the coefficient to zero' by just calculating this regression
        # But then, we manually have to set the residual to the regressand itself #TODO double check if makes sense
        if split_string[0] in split_string[1]:
            if "+" in format_string:
                reduced_format_string = split_string[1].replace("+", "~", 1)
                fit2 = smf.ols(reduced_format_string, data=data).fit()
                return ResidualTriplet(fit1.params[1], data[split_string[0]], fit2.resid)
            else:
                return ResidualTriplet(fit1.params[1], data[split_string[0]], data[split_string[1]])
        # check for empty adjustment set case
        elif "+" in format_string:
            reduced_format_string = split_string[1].replace("+", "~", 1)
            fit2 = smf.ols(reduced_format_string, data=data).fit()
            return ResidualTriplet(fit1.params[1], fit1.resid, fit2.resid)
        else:
            return ResidualTriplet(fit1.params[1], fit1.resid, data[split_string[1]])

    @staticmethod
    def get_est_cov(data: pd.DataFrame, x: int, y: int, separtion_sets: Iterable[Iterable[int]]) -> CovariaceResult:
        var_names = data.columns
        fml = RoCA.get_format_strings(x, y, separtion_sets, var_names)
        res = [RoCA.get_residuals(data, f) for f in fml]
        est = np.array([r.tau for r in res])
        res_y = np.array([r.res_y for r in res])
        res_x = np.array([r.res_x for r in res])
        covmat = np.zeros((len(fml), len(fml)))
        for i in range(len(fml)):
            for j in range(i, len(fml)):
                num = np.mean(res_y[:, i] * res_x[:, i] * res_y[:, j] * res_x[:, j])
                den = np.mean(res_x[:, i] ** 2) * np.mean(res_x[:, j] ** 2)
                covmat[i, j] = covmat[j, i] = num / den
        return CovariaceResult(est, covmat)

    @staticmethod
    def ic_rank_est(sigma: NDArray, n: int) -> int:
        p = sigma.shape[0]
        u, d, _ = np.linalg.svd(sigma)
        ic = np.zeros(p)

        for k in range(1, p + 1):
            if k == 1:
                rec_sigma = np.outer(u[:, 0], u[:, 0]) * d[0]
            else:
                rec_sigma = u[:, :k] @ np.diag(d[:k]) @ u[:, :k].T

            sigma_diff = sigma - rec_sigma
            ic[k - 1] = n * np.sum(sigma_diff[np.tril_indices(p, k=-1)] ** 2) + np.log(n) * (p * k - k * (k - 1) / 2)

        return int(np.argmin(ic))

    def wald_test(self, data: pd.DataFrame, x: int, y: int, separtion_sets: Iterable[Iterable[int]]) -> TestResult:
        est_cov = RoCA.get_est_cov(data=data, x=x, y=y, separtion_sets=separtion_sets)
        n = data.shape[0]
        sigma = est_cov.covmat

        # no. of adjustment sets used
        k = sigma.shape[0]

        if self.contrast is not None:
            if np.any(self.contrast.shape != (k - 1, k)):
                logging.warning(
                    f"Wrong dimension of contrast matrix: expected ({k - 1}, {k}) and got ({self.contrast.shape[0]}, "
                    f"{self.contrast.shape[1]})\nUsing default contrast matrix")
                self.contrast = None
            if not np.allclose(np.sum(self.contrast, axis=0), np.zeros(k - 1)):
                self.contrast = None
                logging.warning("Input matrix is not a valid contrast matrix\nUsing default contrast matrix")

        if self.contrast is None:
            # provide default contrast matrix
            C = np.array(
                [np.concatenate((np.zeros(j - 1), np.array([1, -1]), np.zeros(k - 1 - j))) for j in range(1, k)])
        else:
            C = self.contrast
        v = np.matmul(C, est_cov.est)
        S = np.matmul(np.matmul(C, sigma), C.T)

        if self.rank_est:
            r = RoCA.ic_rank_est(sigma=sigma, n=n)
            if r == 1:
                raise Exception("Rank of Sigma estimated to be one; test not available!")
            r = RoCA.ic_rank_est(sigma=S, n=n)
        else:
            r = k - 1

        u, d, vh = np.linalg.svd(S, full_matrices=False)
        if r > 1:
            s_inv = np.matmul(np.matmul(u[:, :r], np.diag(1 / d[:r])), vh[:r, :])
        else:
            s_inv = 1 / d[0] * np.outer(u[:, 0], vh[0, :])
        csq = n * np.matmul(np.matmul(v.transpose(), s_inv), v)

        pval = chi2.sf(csq, df=r)
        return TestResult(pval, csq, r)

    def test_causal_strength_identical(self, data: pd.DataFrame, x: Any, y: Any,
                                       separation_sets: Iterable[Iterable[Any]],
                                       alpha: float = 0.001) -> bool:
        idx = {n: i for (i, n) in enumerate(data.columns)}
        res = self.wald_test(data, idx[x], idx[y], [[idx[n] for n in sep_s] for sep_s in separation_sets])
        return res.pval > alpha  # Null hyp: they are consistent
