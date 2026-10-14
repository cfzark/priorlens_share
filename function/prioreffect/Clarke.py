import os
import json
import argparse

import numpy as np
import pandas as pd

from itertools import combinations_with_replacement
from math import comb
from joblib import Parallel, delayed

try:
    from .distributions import *
    from .models import *
    from .metrics import *
    from .augmentation import *
    from .utils import *
except ImportError:  # pragma: no cover - keeps old notebook-style imports working
    from distributions import *
    from models import *
    from metrics import *
    from augmentation import *
    from utils import *


class ClarkePriorSizeRunner1D:
    """
    Exact Clarke-style prior size runner.

    For each k:
        d_k^* = min_{S: |S|=k} distance( base-side posterior from subset y_S ,
                                        informative-side prior )

    Then:
        k_hat = argmin_k d_k^*

    Supported logic
    ---------------
    1) One-dimensional comparison:
        KL( posterior_under_baseline || informative_prior )

    2) Joint NIG comparison:
        KL( baseline posterior as NIG || informative NIG prior )

    Notes
    -----
    - For one-dimensional comparisons, `model.posterior(...)` should return
      a univariate Distribution object.
    - For joint unknown-variance Gaussian comparisons, we reconstruct the
      baseline posterior as a NormalInverseGammaDist from posterior parameters.
    """

    def __init__(
        self,
        model,
        prior_p,
        prior_q0,
        distance_type="kl",
        alpha=0.05,
        B=2000,
        B_hdr=20000,
        B_p=20000,
        B_q=20000,
    ):
        self.model = model
        self.prior_p = prior_p          # informative side
        self.prior_q0 = prior_q0        # baseline side
        self.distance_type = distance_type
        self.alpha = alpha
        self.B = B
        self.B_hdr = B_hdr
        self.B_p = B_p
        self.B_q = B_q

        if distance_type == "kl":
            self.distance = KLDivergence(B=B)
        elif distance_type == "truncated_kl":
            self.distance = TruncatedKLDivergence(
                alpha=alpha,
                B=B,
                B_hdr=B_hdr,
                B_p=B_p,
                B_q=B_q,
            )
        elif distance_type == "joint_kl":
            self.distance = KLDivergence(B=B)
        elif distance_type == "joint_truncated_kl":
            self.distance = TruncatedKLDivergence(
                alpha=alpha,
                B=B,
                B_hdr=B_hdr,
                B_p=B_p,
                B_q=B_q,
            )
        else:
            raise ValueError(f"Unsupported distance_type: {distance_type}")

    def _posterior_under_base(self, y_subset):
        """
        Univariate posterior under baseline prior.
        """
        return self.model.posterior(y_subset, self.prior_q0)

    def _joint_posterior_under_base(self, y_subset):
        """
        Reconstruct the baseline posterior as a NormalInverseGammaDist.

        This is valid for NormalMeanUnknownVarModel under:
        - FlatMuInverseGammaPrior
        - JeffreysPrior

        because in both cases the posterior has the form:
            sigma^2 | y ~ IG(kappa_n, eta_n)
            mu | sigma^2, y ~ N(mu_n, sigma^2 / lambda_n)

        which matches:
            mu | sigma^2 ~ N(mu0, tau0_sq * sigma^2)
        with
            tau0_sq = 1 / lambda_n.
        """
        pars = self.model.posterior_params(y_subset, self.prior_q0)

        return NormalInverseGammaDist(
            mu0=pars["mu_loc"],
            tau0_sq=1.0 / pars["lambda_n"],
            kappa=pars["sigma2_shape"],
            eta=pars["sigma2_scale"],
        )

    def _distance_for_subset(self, y_subset, rng=None):
        """
        Return distance for a given subset, depending on model / metric choice.
        """
        # Case A: generic one-dimensional distances
        if self.distance_type in {"kl", "truncated_kl"}:
            post = self._posterior_under_base(y_subset)
            return self.distance.compute(post, self.prior_p, rng=rng)

        # Case B: joint KL for unknown-variance Gaussian
        elif self.distance_type in {"joint_kl", "joint_truncated_kl"}:
            post_joint = self._joint_posterior_under_base(y_subset)
            return self.distance.compute(post_joint, self.prior_p, rng=rng)

        else:
            raise ValueError(f"Unsupported distance_type: {self.distance_type}")

    def run(self, y, k_grid=None, rng=None):
        y = np.asarray(y, dtype=float)
        n = len(y)

        if k_grid is None:
            k_grid = np.arange(1, n + 1)
        else:
            k_grid = np.asarray(k_grid, dtype=int)
            k_grid = k_grid[(k_grid >= 1) & (k_grid <= n)]
            k_grid = np.unique(k_grid)

        if len(k_grid) == 0:
            raise ValueError("k_grid has no valid values in {1, ..., n}.")

        d_star = []
        best_subsets = []
        best_subset_values = []
        n_candidates = []

        for k in k_grid:
            best_d = np.inf
            best_subset = None

            for idx in combinations_with_replacement(range(n), k):
                y_subset = y[list(idx)]

                try:
                    d_val = self._distance_for_subset(y_subset, rng=rng)
                except Exception:
                    # e.g. improper posterior for very small subset,
                    # or S=0 under Jeffreys, etc.
                    continue

                if np.isnan(d_val) or np.isinf(d_val):
                    continue

                if d_val < best_d:
                    best_d = d_val
                    best_subset = idx

            d_star.append(best_d)
            best_subsets.append(best_subset)

            if best_subset is None:
                best_subset_values.append(None)
            else:
                best_subset_values.append(y[list(best_subset)].tolist())

            n_candidates.append(comb(n + k - 1, k))

        d_star = np.asarray(d_star, dtype=float)

        finite_mask = np.isfinite(d_star)
        if not np.any(finite_mask):
            raise RuntimeError(
                "No valid subset produced a finite distance. "
                "Check model/prior/metric settings."
            )

        argmin_idx = int(np.nanargmin(d_star))
        k_hat = int(k_grid[argmin_idx])

        return {
            "k_grid": np.asarray(k_grid, dtype=int),
            "d_star": d_star,
            "k_hat": k_hat,
            "best_subsets": best_subsets,
            "best_subset_values": best_subset_values,
            "selected_subset_idx": best_subsets[argmin_idx],
            "selected_subset_values": best_subset_values[argmin_idx],
            "n_candidates": np.asarray(n_candidates, dtype=int),
        }


def build_model_and_priors(args, prior_loc):
    """
    Build model, informative prior, and baseline prior for a given prior location/mean.

    Convention used here
    --------------------
    We always compare:

        baseline posterior || informative prior

    That is:
    - 1D marginal comparison:
        KL( posterior_under_baseline_on_parameter_of_interest || informative_prior_marginal )

    - Joint Gaussian unknown-variance comparison:
        KL( baseline_posterior_as_NIG || informative_NIG_prior )
    """
    # -------------------------------------------------
    # 1) Gaussian mean known variance
    # -------------------------------------------------
    if args.model_type == "normal_known":
        model = NormalMeanKnownVarModel(sigma=args.sigma)

        # informative prior on mu
        prior_p = NormalDist(prior_loc, args.prior_sd)

        # baseline prior on mu
        if args.baseline_type == "flat_mu":
            prior_q0 = FlatPrior()
        elif args.baseline_type == "jeffreys_mu":
            prior_q0 = JeffreysPrior()
        else:
            raise ValueError(
                "For normal_known, baseline_type must be 'flat_mu' or 'jeffreys_mu'."
            )

        allowed_metrics = {"kl", "truncated_kl"}

    # -------------------------------------------------
    # 2) Gaussian mean unknown variance: compare mu marginal
    # -------------------------------------------------
    elif args.model_type == "normal_unknown_marginal":
        model = NormalMeanUnknownVarModel()

        informative_nig = NormalInverseGammaDist(
            mu0=prior_loc,
            tau0_sq=args.tau0_sq,
            kappa=args.kappa,
            eta=args.eta
        )

        # informative side should be marginal prior of mu
        prior_p = StudentTDist(
            df=2.0 * informative_nig.kappa,
            loc=informative_nig.mu0,
            scale=np.sqrt(informative_nig.eta * informative_nig.tau0_sq / informative_nig.kappa)
        )

        if args.baseline_type == "flat_mu_ig":
            prior_q0 = FlatMuInverseGammaPrior(kappa=args.kappa, eta=args.eta)
        elif args.baseline_type == "jeffreys_mu_sigma2":
            prior_q0 = JeffreysPrior()
        else:
            raise ValueError(
                "For normal_unknown_marginal, baseline_type must be "
                "'flat_mu_ig' or 'jeffreys_mu_sigma2'."
            )

        allowed_metrics = {"kl", "truncated_kl"}

    # -------------------------------------------------
    # 3) Gaussian mean unknown variance: compare joint NIG
    # -------------------------------------------------
    elif args.model_type == "normal_unknown_joint":
        model = NormalMeanUnknownVarModel()

        prior_p = NormalInverseGammaDist(
            mu0=prior_loc,
            tau0_sq=args.tau0_sq,
            kappa=args.kappa,
            eta=args.eta
        )

        if args.baseline_type == "flat_mu_ig":
            prior_q0 = FlatMuInverseGammaPrior(kappa=args.kappa, eta=args.eta)
        elif args.baseline_type == "jeffreys_mu_sigma2":
            prior_q0 = JeffreysPrior()
        else:
            raise ValueError(
                "For normal_unknown_joint, baseline_type must be "
                "'flat_mu_ig' or 'jeffreys_mu_sigma2'."
            )

        allowed_metrics = {"joint_kl", "joint_truncated_kl"}

    # -------------------------------------------------
    # 4) Poisson
    # -------------------------------------------------
    elif args.model_type == "poisson":
        model = PoissonModel()

        # informative Gamma prior parameterized by mean + strength
        # mean = alpha / beta, choose beta = strength, alpha = mean * strength
        alpha0 = prior_loc * args.poisson_prior_strength
        beta0 = args.poisson_prior_strength

        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError(
                "Poisson informative Gamma prior requires positive alpha and beta."
            )

        # your current PoissonModel uses GammaDist directly as informative prior
        prior_p = GammaDist(alpha0, beta0)

        if args.baseline_type == "flat_lambda":
            prior_q0 = FlatPrior()
        elif args.baseline_type == "jeffreys_lambda":
            prior_q0 = JeffreysPrior()
        else:
            raise ValueError(
                "For poisson, baseline_type must be 'flat_lambda' or 'jeffreys_lambda'."
            )

        allowed_metrics = {"kl", "truncated_kl"}

    # -------------------------------------------------
    # 5) Beta-Binomial
    # -------------------------------------------------
    elif args.model_type == "beta_binomial":
        model = BinomialModel(trials=args.binomial_trials)

        alpha0 = prior_loc * args.beta_binomial_prior_strength
        beta0 = (1.0 - prior_loc) * args.beta_binomial_prior_strength

        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError(
                "Beta-Binomial informative Beta prior requires prior_loc in (0, 1) "
                "and positive prior strength."
            )

        prior_p = BetaDist(alpha0, beta0)

        if args.baseline_type == "flat_prob":
            prior_q0 = FlatPrior()
        elif args.baseline_type == "jeffreys_prob":
            prior_q0 = JeffreysPrior()
        else:
            raise ValueError(
                "For beta_binomial, baseline_type must be 'flat_prob' or 'jeffreys_prob'."
            )

        allowed_metrics = {"kl", "truncated_kl"}

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if args.distance_type not in allowed_metrics:
        raise ValueError(
            f"distance_type='{args.distance_type}' is not allowed for model_type='{args.model_type}'. "
            f"Allowed: {sorted(allowed_metrics)}"
        )

    return model, prior_p, prior_q0


def simulate_one_dataset(args, rng):
    """
    Simulate one dataset according to model_type.
    """
    if args.model_type in {
        "normal_known",
        "normal_unknown_marginal",
        "normal_unknown_joint",
    }:
        y = rng.normal(loc=args.true_mu, scale=args.true_sigma, size=args.n)
        return y.astype(float)

    elif args.model_type == "poisson":
        y = rng.poisson(lam=args.true_lambda, size=args.n)
        return y.astype(float)

    elif args.model_type == "beta_binomial":
        y = rng.binomial(n=args.binomial_trials, p=args.true_p, size=args.n)
        return y.astype(float)

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")


def compute_one_ess(prior_loc, r, args):
    """
    For fixed replicate r, all prior locations share the same generated sample.
    """
    rng = np.random.default_rng(args.base_seed + r)

    y_rep = simulate_one_dataset(args, rng)
    model, prior_p, prior_q0 = build_model_and_priors(args, prior_loc)

    k_grid = np.asarray(args.m_grid, dtype=int)
    k_grid = k_grid[(k_grid >= 1) & (k_grid <= args.n)]
    k_grid = np.unique(k_grid)

    runner = ClarkePriorSizeRunner1D(
        model=model,
        prior_p=prior_p,
        prior_q0=prior_q0,
        distance_type=args.distance_type,
        alpha=args.alpha,
        B=args.mc_B,
        B_hdr=args.B_hdr,
        B_p=args.B_p,
        B_q=args.B_q,
    )

    out = runner.run(y_rep, k_grid=k_grid, rng=rng)
    min_kl = float(np.nanmin(out["d_star"]))

    return {
        "rep": int(r),
        "prior_loc": float(prior_loc),
        "sample": y_rep.tolist(),
        "selected_sample": out["selected_subset_values"],
        "ess": int(out["k_hat"]),
        "min_kl": min_kl,
        "distance_type": args.distance_type,
        "model_type": args.model_type,
        "baseline_type": args.baseline_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Clarke-style ESS simulation for multiple model/prior settings."
    )

    # ----------------------------
    # General
    # ----------------------------
    parser.add_argument("--R", type=int, default=50, help="Number of replicates")
    parser.add_argument(
        "--rep_start",
        type=int,
        default=0,
        help="Start replicate index (inclusive)."
    )
    parser.add_argument(
        "--rep_end",
        type=int,
        default=None,
        help="End replicate index (exclusive). Default: R."
    )
    parser.add_argument("--n", type=int, default=20, help="Sample size per replicate")
    parser.add_argument("--base_seed", type=int, default=321, help="Base random seed")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--save_dir", type=str, default="results_clarke", help="Directory to save results")

    parser.add_argument(
        "--model_type",
        type=str,
        default="normal_known",
        choices=[
            "normal_known",
            "normal_unknown_marginal",
            "normal_unknown_joint",
            "poisson",
            "beta_binomial",
        ],
        help="Which model setting to run",
    )

    parser.add_argument(
        "--distance_type",
        type=str,
        default="kl",
        choices=[
            "kl",
            "truncated_kl",
            "joint_kl",
            "joint_truncated_kl",
        ],
        help="Distance metric type",
    )

    parser.add_argument("--alpha", type=float, default=0.05, help="Truncation level alpha")
    parser.add_argument("--mc_B", type=int, default=2000, help="MC size for generic KL / truncated KL")
    parser.add_argument("--B_hdr", type=int, default=5000, help="MC size for HDR threshold estimation")
    parser.add_argument("--B_p", type=int, default=5000, help="MC size for truncated joint KL core")
    parser.add_argument("--B_q", type=int, default=5000, help="MC size for truncated joint KL region probability")

    parser.add_argument(
        "--prior_locs",
        type=float,
        nargs="+",
        default=[0, 1, 3, 5],
        help=(
            "Grid of informative prior locations/means. "
            "For Gaussian models this is mu0; for Poisson this is prior mean of lambda."
        ),
    )

    parser.add_argument(
        "--m_grid",
        type=int,
        nargs="+",
        default=list(range(1, 10)),
        help="Candidate ESS values",
    )

    # ----------------------------
    # Gaussian known variance
    # ----------------------------
    parser.add_argument("--true_mu", type=float, default=0.0, help="True mean for Gaussian data generation")
    parser.add_argument("--true_sigma", type=float, default=np.sqrt(3.0), help="True sigma for Gaussian data generation")
    parser.add_argument("--sigma", type=float, default=np.sqrt(3.0), help="Known sigma used in normal_known model")
    parser.add_argument("--prior_sd", type=float, default=float(np.sqrt(0.2)), help="Informative prior sd for normal_known")

    # ----------------------------
    # Gaussian unknown variance (NIG prior)
    # ----------------------------
    parser.add_argument("--tau0_sq", type=float, default=0.5, help="tau0^2 in NIG prior: Var(mu|sigma^2)=tau0_sq*sigma^2")
    parser.add_argument("--kappa", type=float, default=2.0, help="Inverse-Gamma shape in NIG prior")
    parser.add_argument("--eta", type=float, default=3.0, help="Inverse-Gamma scale in NIG prior")

    parser.add_argument(
        "--baseline_type",
        type=str,
        default="flat_mu",
        choices=[
            "flat_mu",            # for normal_known
            "jeffreys_mu",        # for normal_known
            "flat_mu_ig",         # for normal_unknown_*
            "jeffreys_mu_sigma2", # for normal_unknown_*
            "flat_lambda",        # for poisson
            "jeffreys_lambda",    # for poisson
            "flat_prob",          # for beta_binomial
            "jeffreys_prob",      # for beta_binomial
        ],
        help="Baseline prior type, interpreted according to model_type",
    )

    # ----------------------------
    # Poisson
    # ----------------------------
    parser.add_argument("--true_lambda", type=float, default=3.0, help="True lambda for Poisson data generation")
    parser.add_argument("--poisson_prior_strength", type=float, default=2.0, help="Gamma prior strength beta; alpha = prior_mean * strength")

    # ----------------------------
    # Beta-Binomial
    # ----------------------------
    parser.add_argument("--true_p", type=float, default=0.3, help="True success probability for Binomial data generation")
    parser.add_argument("--binomial_trials", type=int, default=20, help="Number of Binomial trials per observation")
    parser.add_argument(
        "--beta_binomial_prior_strength",
        type=float,
        default=80.0,
        help=(
            "Beta prior strength alpha + beta in Bernoulli-trial units. "
            "With binomial_trials=20, 80 corresponds to nominal ESS=4 Binomial observations."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.rep_end is None:
        args.rep_end = args.R

    if not (0 <= args.rep_start < args.rep_end <= args.R):
        raise ValueError(
            f"Require 0 <= rep_start < rep_end <= R, but got "
            f"rep_start={args.rep_start}, rep_end={args.rep_end}, R={args.R}."
        )

    os.makedirs(args.save_dir, exist_ok=True)

    rep_range = range(args.rep_start, args.rep_end)
    tasks = [(prior_loc, r) for r in rep_range for prior_loc in args.prior_locs]

    results_list = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(compute_one_ess)(
            prior_loc=prior_loc,
            r=r,
            args=args
        )
        for prior_loc, r in tasks
    )

    # save detailed results
    json_name = f"all_results.json"
    json_path = os.path.join(args.save_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)

    # summary
    df = pd.DataFrame(results_list)
    summary_df = (
        df.groupby("prior_loc")["ess"]
        .agg(mean_ess="mean", sd_ess="std")
        .reset_index()
    )

    txt_name = f"summary_rep_{args.rep_start}_{args.rep_end}.txt"
    txt_path = os.path.join(args.save_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"model_type = {args.model_type}\n")
        f.write(f"distance_type = {args.distance_type}\n")
        f.write(f"baseline_type = {args.baseline_type}\n")
        f.write(f"rep_start = {args.rep_start}\n")
        f.write(f"rep_end = {args.rep_end}\n\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"prior_loc={row['prior_loc']}, "
                f"mean_ess={row['mean_ess']:.6f}, "
                f"sd_ess={row['sd_ess']:.6f}\n"
            )

    print("Finished.")
    print(f"Replications covered: [{args.rep_start}, {args.rep_end})")
    print(f"Detailed results saved to: {json_path}")
    print(f"Summary TXT saved to: {txt_path}")


if __name__ == "__main__":
    main()
