from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
try:
    from .distributions import *
    from .utils import obj_func
except ImportError:  # pragma: no cover - keeps old notebook-style imports working
    from distributions import *
    from utils import obj_func
from scipy import integrate
from scipy.special import gammaln, digamma, erf
from scipy import stats


# Distance metrics
class DistanceMetric(ABC):
    @abstractmethod
    def compute(self, P, Q, rng=None):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class KLDivergence(DistanceMetric):
    """
    Standard KL(P || Q)

    Supported special cases:
    - NormalDist vs NormalDist: closed form
    - StudentTDist vs StudentTDist: 1D numerical expectation
    - InverseGammaDist vs InverseGammaDist: closed form

    Otherwise: Monte Carlo fallback
    """
    def __init__(self, B=5000, epsabs=1e-10, epsrel=1e-8):
        self.B = B
        self.epsabs = epsabs
        self.epsrel = epsrel

    def __repr__(self):
        return f"KLDivergence(B={self.B}, epsabs={self.epsabs}, epsrel={self.epsrel})"

    def compute(self, P, Q, rng=None):
        rng = np.random.default_rng() if rng is None else rng

        if isinstance(P, NormalDist) and isinstance(Q, NormalDist):
            return self._kl_normal_normal(P, Q)

        if isinstance(P, StudentTDist) and isinstance(Q, StudentTDist):
            return self._kl_student_t_student_t(P, Q)

        if isinstance(P, InverseGammaDist) and isinstance(Q, InverseGammaDist):
            return self._kl_invgamma_invgamma(P, Q)
        
        if isinstance(P, GammaDist) and isinstance(Q, GammaDist):
            return self._kl_gamma_gamma(P, Q)

        if isinstance(P, BetaDist) and isinstance(Q, BetaDist):
            return self._kl_beta_beta(P, Q)
        
        if isinstance(P, NormalInverseGammaDist) and isinstance(Q, NormalInverseGammaDist):
            return self._kl_nig_nig(P, Q)

        x = P.sample(self.B, rng=rng)
        return float(np.mean(P.logpdf(x) - Q.logpdf(x)))

    @staticmethod
    def _kl_normal_normal(P, Q):
        mu1, sd1 = P.mean(), np.sqrt(P.var())
        mu2, sd2 = Q.mean(), np.sqrt(Q.var())
        return float(
            np.log(sd2 / sd1)
            + (sd1**2 + (mu1 - mu2)**2) / (2 * sd2**2)
            - 0.5
        )

    @staticmethod
    def _kl_invgamma_invgamma(P, Q):
        """
        P = IG(a1, b1), Q = IG(a2, b2)
        density: f(x)=b^a/Gamma(a) * x^{-a-1} exp(-b/x)

        KL(P||Q) =
            logGamma(a2) - logGamma(a1)
            + a2(log b1 - log b2)
            + (a1-a2) psi(a1)
            + a1 (b2-b1)/b1
        """
        a1, b1 = P.shape, P.scale
        a2, b2 = Q.shape, Q.scale

        out = (
            gammaln(a2) - gammaln(a1)
            + a2 * (np.log(b1) - np.log(b2))
            + (a1 - a2) * digamma(a1)
            + a1 * (b2 - b1) / b1
        )
        return float(out)
    
    @staticmethod
    def _kl_gamma_gamma(P, Q):
        """
        P = Gamma(alpha1, beta1), Q = Gamma(alpha2, beta2)
        shape-rate parameterization

        KL(P||Q) =
            alpha2 * log(beta1/beta2)
            - logGamma(alpha1) + logGamma(alpha2)
            + (alpha1 - alpha2) * digamma(alpha1)
            + alpha1 * (beta2/beta1 - 1)
        """
        a1, b1 = P.alpha, P.beta
        a2, b2 = Q.alpha, Q.beta

        out = (
            a2 * np.log(b1 / b2)
            - gammaln(a1) + gammaln(a2)
            + (a1 - a2) * digamma(a1)
            + a1 * (b2 / b1 - 1.0)
        )
        return float(out)

    @staticmethod
    def _kl_beta_beta(P, Q):
        """
        KL(P || Q) for Beta(alpha, beta) distributions.
        """
        a1, b1 = P.alpha, P.beta
        a2, b2 = Q.alpha, Q.beta

        out = (
            gammaln(a2) + gammaln(b2) - gammaln(a2 + b2)
            - gammaln(a1) - gammaln(b1) + gammaln(a1 + b1)
            + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1)
            - (a1 + b1 - a2 - b2) * digamma(a1 + b1)
        )
        return float(out)

    def _kl_student_t_student_t(self, P, Q):
        """
        KL(P || Q) for two location-scale Student t distributions.

        If
            P = t_{nu1}(m1, s1),
            Q = t_{nu2}(m2, s2),
        then

            KL(P||Q)
            = C
            - (nu1+1)/2 * E_{Z~t_{nu1}}[ log(1 + Z^2 / nu1) ]
            + (nu2+1)/2 * E_{Z~t_{nu1}}[
                    log(1 + (delta + r Z)^2 / nu2)
                ]

        where
            delta = (m1 - m2) / s2,
            r     = s1 / s2.

        The first expectation has closed form:
            E[log(1 + Z^2/nu1)] = psi((nu1+1)/2) - psi(nu1/2).

        So only the second expectation is evaluated numerically.
        """
        nu1, m1, s1 = float(P.df), float(P.loc), float(P.scale)
        nu2, m2, s2 = float(Q.df), float(Q.loc), float(Q.scale)

        if nu1 <= 0 or nu2 <= 0:
            raise ValueError("Student t degrees of freedom must be positive.")
        if s1 <= 0 or s2 <= 0:
            raise ValueError("Student t scales must be positive.")

        if (
            np.isclose(nu1, nu2)
            and np.isclose(m1, m2)
            and np.isclose(s1, s2)
        ):
            return 0.0

        # delta and relative scale
        delta = (m1 - m2) / s2
        r = s1 / s2

        # log normalizing constants:
        # c(nu, s) = log Gamma((nu+1)/2) - log Gamma(nu/2)
        #            - 0.5 log(nu*pi) - log s
        c1 = (
            gammaln((nu1 + 1.0) / 2.0)
            - gammaln(nu1 / 2.0)
            - 0.5 * np.log(nu1 * np.pi)
            - np.log(s1)
        )
        c2 = (
            gammaln((nu2 + 1.0) / 2.0)
            - gammaln(nu2 / 2.0)
            - 0.5 * np.log(nu2 * np.pi)
            - np.log(s2)
        )

        const_term = c1 - c2

        # closed-form first expectation
        e1 = digamma((nu1 + 1.0) / 2.0) - digamma(nu1 / 2.0)

        # density of standard t_{nu1}
        log_c_std1 = (
            gammaln((nu1 + 1.0) / 2.0)
            - gammaln(nu1 / 2.0)
            - 0.5 * np.log(nu1 * np.pi)
        )

        def f_std_t_nu1(z):
            return np.exp(log_c_std1 - ((nu1 + 1.0) / 2.0) * np.log1p(z * z / nu1))

        # only remaining 1D expectation
        def integrand(z):
            return np.log1p(((delta + r * z) ** 2) / nu2) * f_std_t_nu1(z)

        e2, _ = integrate.quad(
            integrand,
            -np.inf,
            np.inf,
            epsabs=self.epsabs,
            epsrel=self.epsrel,
            limit=200
        )

        kl = const_term - ((nu1 + 1.0) / 2.0) * e1 + ((nu2 + 1.0) / 2.0) * e2
        return float(kl)
    
    @staticmethod
    def _kl_nig_nig(P, Q):
        """
        KL(P || Q) for two Normal-Inverse-Gamma distributions.

        Parameterization:
            sigma^2 ~ IG(kappa, eta)
            mu | sigma^2 ~ N(mu0, tau0_sq * sigma^2)

        where IG density is
            f(x) = eta^kappa / Gamma(kappa) * x^{-kappa-1} exp(-eta/x), x > 0
        """
        mu1, tau1_sq, kappa1, eta1 = P.mu0, P.tau0_sq, P.kappa, P.eta
        mu2, tau2_sq, kappa2, eta2 = Q.mu0, Q.tau0_sq, Q.kappa, Q.eta

        if tau1_sq <= 0 or tau2_sq <= 0:
            raise ValueError("tau0_sq must be positive.")
        if kappa1 <= 0 or kappa2 <= 0:
            raise ValueError("kappa must be positive.")
        if eta1 <= 0 or eta2 <= 0:
            raise ValueError("eta must be positive.")

        # KL between IG marginals
        kl_sigma2 = (
            gammaln(kappa2) - gammaln(kappa1)
            + kappa2 * (np.log(eta1) - np.log(eta2))
            + (kappa1 - kappa2) * digamma(kappa1)
            + kappa1 * (eta2 - eta1) / eta1
        )

        # E_P[1/sigma^2] under IG(kappa1, eta1)
        e_inv_sigma2 = kappa1 / eta1

        # E_P KL( N(mu1, tau1_sq*sigma^2) || N(mu2, tau2_sq*sigma^2) )
        kl_mu_given_sigma2 = 0.5 * (
            np.log(tau2_sq / tau1_sq)
            + (tau1_sq / tau2_sq)
            - 1.0
            + ((mu1 - mu2) ** 2 / tau2_sq) * e_inv_sigma2
        )

        return float(kl_sigma2 + kl_mu_given_sigma2)


class TruncatedKLDivergence(DistanceMetric):
    """
    Truncated KL_alpha(P || Q)

    - closed-form for NormalDist vs NormalDist
    - generic MC version otherwise
    """
    def __init__(self, alpha=0.05, B=2000, eps=1e-12,
                 B_hdr=10000, B_p=10000, B_q=10000):
        self.alpha = alpha
        self.B = B
        self.eps = eps
        self.B_hdr = B_hdr
        self.B_p = B_p
        self.B_q = B_q

    def __repr__(self):
        return f"TruncatedKLDivergence(alpha={self.alpha}, B={self.B}, B_hdr={self.B_hdr}, B_p={self.B_p}, B_q={self.B_q})"

    def compute(self, P, Q, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        alpha = self.alpha

        if isinstance(P, NormalDist) and isinstance(Q, NormalDist):
            mu1, sd1 = P.mean(), np.sqrt(P.var())
            mu2, sd2 = Q.mean(), np.sqrt(Q.var())

            a = norm.ppf(alpha / 2, loc=mu1, scale=sd1)
            b = norm.ppf(1 - alpha / 2, loc=mu1, scale=sd1)

            za = (a - mu1) / sd1
            zb = (b - mu1) / sd1

            Phi = norm.cdf
            phi = norm.pdf

            Pab = Phi(zb) - Phi(za)

            EX = mu1 + sd1 * (phi(za) - phi(zb)) / Pab

            EX2 = (
                mu1**2
                + 2 * mu1 * sd1 * (phi(za) - phi(zb)) / Pab
                + sd1**2 * (1 + (za * phi(za) - zb * phi(zb)) / Pab)
            )

            EXmu2 = EX2 - 2 * mu2 * EX + mu2**2
            EXmu1 = EX2 - 2 * mu1 * EX + mu1**2

            core = np.log(sd2 / sd1) + EXmu2 / (2 * sd2**2) - EXmu1 / (2 * sd1**2)

            Qab = norm.cdf((b - mu2) / sd2) - norm.cdf((a - mu2) / sd2)

            return float(core + np.log(Qab / Pab))

        if isinstance(P, NormalInverseGammaDist) and isinstance(Q, NormalInverseGammaDist):
            return self.truncated_kl_nig_nig_hdr(
                P, Q,
                alpha=self.alpha,
                B_hdr=self.B_hdr,
                B_p=self.B_p,
                B_q=self.B_q,
                eps=self.eps,
                rng=rng,
                return_details=False
            )
        
        x = P.sample(self.B, rng=rng)

        if alpha > 0:
            a, b = np.quantile(x, [alpha / 2, 1 - alpha / 2])
            xI = x[(x >= a) & (x <= b)]
            if len(xI) == 0:
                return np.nan
        else:
            a, b = None, None
            xI = x

        log_ratio = P.logpdf(xI) - Q.logpdf(xI)

        if alpha <= 0:
            return float(np.mean(log_ratio))
        else:
            QI = Q.prob_interval(a, b)
            QI = max(QI, self.eps)
            return float(np.mean(log_ratio) + np.log(QI / (1 - alpha)))
        
    @staticmethod
    def truncated_kl_nig_nig_hdr(
        P,
        Q,
        alpha=0.05,
        B_hdr=10000,
        B_p=10000,
        B_q=10000,
        eps=1e-12,
        rng=None,
        return_details=False
    ):
        """
        Monte Carlo truncated KL using an HDR region based on P.

        We estimate
            D_tr(P || Q; A_alpha)
        = E_{P_A}[ log p(Z) - log q(Z) ] + log( Q(A_alpha) / P(A_alpha) )

        where
            A_alpha = { z : log p(z) >= c_alpha }
        and c_alpha is chosen so that approximately P(A_alpha) = 1 - alpha.

        Parameters
        ----------
        P, Q : NormalInverseGammaDist
        alpha : float
            Truncation level; nominal P-mass retained is about 1 - alpha.
        B_hdr : int
            Number of P-samples used to estimate the HDR threshold.
        B_p : int
            Number of fresh P-samples used to estimate the truncated expectation.
        B_q : int
            Number of Q-samples used to estimate Q(A_alpha).
        eps : float
            Numerical floor for estimated probabilities.
        rng : numpy Generator or None
        return_details : bool

        Returns
        -------
        float
            Estimated truncated KL.
        or
        (float, dict)
            If return_details=True.
        """
        if not isinstance(P, NormalInverseGammaDist) or not isinstance(Q, NormalInverseGammaDist):
            raise ValueError("P and Q must both be NormalInverseGammaDist.")

        if not (0.0 <= alpha < 1.0):
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")
        if B_hdr <= 0 or B_p <= 0 or B_q <= 0:
            raise ValueError("B_hdr, B_p, B_q must be positive integers.")

        rng = np.random.default_rng() if rng is None else rng

        # ------------------------------------------------------------
        # Step 1: estimate HDR threshold c_alpha from P
        # ------------------------------------------------------------
        mu_hdr, sigma2_hdr = P.sample(size=B_hdr, rng=rng)
        logp_hdr = P.logpdf(mu_hdr, sigma2_hdr)

        # alpha-quantile of log-density => keeps approx top (1-alpha) mass
        c_alpha = np.quantile(logp_hdr, alpha)

        # ------------------------------------------------------------
        # Step 2: estimate conditional expectation under P
        # ------------------------------------------------------------
        mu_p, sigma2_p = P.sample(size=B_p, rng=rng)
        logp_p = P.logpdf(mu_p, sigma2_p)
        logq_p = Q.logpdf(mu_p, sigma2_p)

        mask_p = logp_p >= c_alpha
        pA_hat = np.mean(mask_p)

        if np.sum(mask_p) == 0:
            raise RuntimeError("No P-samples fell inside the estimated HDR region. Increase B_hdr/B_p.")

        core_hat = np.mean(logp_p[mask_p] - logq_p[mask_p])

        # ------------------------------------------------------------
        # Step 3: estimate Q(A_alpha)
        # ------------------------------------------------------------
        mu_q, sigma2_q = Q.sample(size=B_q, rng=rng)
        logp_q = P.logpdf(mu_q, sigma2_q)

        mask_q = logp_q >= c_alpha
        qA_hat = np.mean(mask_q)

        pA_hat = max(float(pA_hat), eps)
        qA_hat = max(float(qA_hat), eps)

        d_trunc = core_hat + np.log(qA_hat / pA_hat)

        if return_details:
            details = {
                "alpha": float(alpha),
                "c_alpha": float(c_alpha),
                "pA_hat": float(pA_hat),
                "qA_hat": float(qA_hat),
                "n_in_region_P": int(np.sum(mask_p)),
                "n_in_region_Q": int(np.sum(mask_q)),
                "core_hat": float(core_hat),
                "B_hdr": int(B_hdr),
                "B_p": int(B_p),
                "B_q": int(B_q),
            }
            return float(d_trunc), details

        return float(d_trunc)


class MeanDifference(DistanceMetric):
    def compute(self, P, Q, rng=None):
        return float(abs(P.mean() - Q.mean()))


class VarianceDifference(DistanceMetric):
    def compute(self, P, Q, rng=None):
        return float(abs(P.var() - Q.var()))


class Wasserstein1Normal(DistanceMetric):
    """
    Exact W1 distance for two univariate normal distributions.
    """
    def compute(self, P, Q, rng=None):
        if not isinstance(P, NormalDist) or not isinstance(Q, NormalDist):
            raise ValueError("Wasserstein1Normal requires both P and Q to be NormalDist.")

        m1, s1 = P.mean(), np.sqrt(P.var())
        m2, s2 = Q.mean(), np.sqrt(Q.var())

        a = float(m1 - m2)
        b = float(s1 - s2)

        if np.isclose(b, 0.0):
            return float(abs(a))

        z = abs(a) / abs(b)

        return float(
            abs(b) * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z**2)
            + abs(a) * (1.0 - 2.0 * norm.cdf(-z))
        )


class ObjectiveFunctionMetric(DistanceMetric):
    """
    Reuse your cp/objective function for approximately normal posteriors.
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def compute(self, P, Q, rng=None):
        mA, vA = P.mean(), P.var()
        mB, vB = Q.mean(), Q.var()
        CP1, CP2, r1, r2, obj, ciA, ciB = obj_func(mA, vA, mB, vB, alpha=self.alpha)
        return float(obj)

class WassersteinDistance(DistanceMetric):
    """
    Wasserstein distance.

    Currently supported:
    - NormalDist vs NormalDist:
        * order = 2 : closed-form W2
        * order = 1 : closed-form W1
    - GammaDist vs GammaDist:
        * order = 2 : numerical quantile integration
        * order = 1 : numerical quantile integration
    - BetaDist vs BetaDist:
        * order = 2 : numerical quantile integration
        * order = 1 : numerical quantile integration

    Notes
    -----
    For one-dimensional distributions:
        W_p^p(P,Q) = int_0^1 |F_P^{-1}(u)-F_Q^{-1}(u)|^p du.

    For GammaDist, we use scipy.stats.gamma.ppf with scale = 1 / beta
    because your GammaDist uses shape-rate parameterization.
    """
    def __init__(
        self,
        order=2,
        gamma_method="grid",
        gamma_grid_size=2000,
        eps=1e-6,
        epsabs=1e-8,
        epsrel=1e-6,
    ):
        if order not in {1, 2}:
            raise ValueError("Currently only order=1 or order=2 is supported.")
        if gamma_method not in {"grid", "quad"}:
            raise ValueError("gamma_method must be 'grid' or 'quad'.")

        self.order = int(order)
        self.gamma_method = gamma_method
        self.gamma_grid_size = int(gamma_grid_size)
        self.eps = float(eps)
        self.epsabs = float(epsabs)
        self.epsrel = float(epsrel)

    def __repr__(self):
        return (
            f"WassersteinDistance(order={self.order}, "
            f"gamma_method='{self.gamma_method}', "
            f"gamma_grid_size={self.gamma_grid_size})"
        )

    def compute(self, P, Q, rng=None):
        if isinstance(P, NormalDist) and isinstance(Q, NormalDist):
            if self.order == 2:
                return self._w2_normal_normal(P, Q)
            elif self.order == 1:
                return self._w1_normal_normal(P, Q)

        if isinstance(P, StudentTDist) and isinstance(Q, StudentTDist):
            return self._wp_student_t_student_t(P, Q)

        if isinstance(P, GammaDist) and isinstance(Q, GammaDist):
            return self._wp_gamma_gamma(P, Q)

        if isinstance(P, BetaDist) and isinstance(Q, BetaDist):
            return self._wp_beta_beta(P, Q)

        raise NotImplementedError(
            "WassersteinDistance currently supports only "
            "NormalDist vs NormalDist and GammaDist vs GammaDist."
        )

    @staticmethod
    def _w2_normal_normal(P, Q):
        mu1, sd1 = P.mean(), np.sqrt(P.var())
        mu2, sd2 = Q.mean(), np.sqrt(Q.var())

        return float(np.sqrt((mu1 - mu2) ** 2 + (sd1 - sd2) ** 2))

    @staticmethod
    def _w1_normal_normal(P, Q):
        from scipy.special import erf

        mu1, sd1 = P.mean(), np.sqrt(P.var())
        mu2, sd2 = Q.mean(), np.sqrt(Q.var())

        delta = float(mu1 - mu2)
        tau = float(abs(sd1 - sd2))

        if np.isclose(tau, 0.0):
            return float(abs(delta))

        z = abs(delta) / tau
        val = (
            tau * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z * z)
            + abs(delta) * erf(z / np.sqrt(2.0))
        )
        return float(val)

    def _gamma_ppf(self, u, G):
        """
        GammaDist uses shape-rate:
            Gamma(alpha, beta)
        scipy uses shape-scale:
            scale = 1 / beta
        """
        return stats.gamma.ppf(u, a=G.alpha, scale=1.0 / G.beta)

    @staticmethod
    def _student_t_ppf(u, T):
        return stats.t.ppf(u, df=T.df, loc=T.loc, scale=T.scale)

    def _wp_student_t_student_t(self, P, Q):
        p = self.order
        u = np.linspace(self.eps, 1.0 - self.eps, self.gamma_grid_size)
        qP = self._student_t_ppf(u, P)
        qQ = self._student_t_ppf(u, Q)
        integrand = np.abs(qP - qQ) ** p
        val_p = np.trapz(integrand, u)
        return float(val_p ** (1.0 / p))

    def _wp_gamma_gamma(self, P, Q):
        p = self.order

        if self.gamma_method == "grid":
            # Avoid exactly 0 and 1 because ppf(0)=0 and ppf(1)=inf.
            u = np.linspace(self.eps, 1.0 - self.eps, self.gamma_grid_size)

            qP = self._gamma_ppf(u, P)
            qQ = self._gamma_ppf(u, Q)

            integrand = np.abs(qP - qQ) ** p
            val_p = np.trapz(integrand, u)

            return float(val_p ** (1.0 / p))

        elif self.gamma_method == "quad":
            def integrand(u):
                qP = self._gamma_ppf(u, P)
                qQ = self._gamma_ppf(u, Q)
                return abs(qP - qQ) ** p

            val_p, _ = integrate.quad(
                integrand,
                self.eps,
                1.0 - self.eps,
                epsabs=self.epsabs,
                epsrel=self.epsrel,
                limit=200
            )

            return float(val_p ** (1.0 / p))

        else:
            raise ValueError("Unknown gamma_method.")

    def _beta_ppf(self, u, B):
        return stats.beta.ppf(u, a=B.alpha, b=B.beta)

    def _wp_beta_beta(self, P, Q):
        p = self.order

        if self.gamma_method == "grid":
            u = np.linspace(self.eps, 1.0 - self.eps, self.gamma_grid_size)
            qP = self._beta_ppf(u, P)
            qQ = self._beta_ppf(u, Q)
            integrand = np.abs(qP - qQ) ** p
            val_p = np.trapz(integrand, u)
            return float(val_p ** (1.0 / p))

        elif self.gamma_method == "quad":
            def integrand(u):
                qP = self._beta_ppf(u, P)
                qQ = self._beta_ppf(u, Q)
                return abs(qP - qQ) ** p

            val_p, _ = integrate.quad(
                integrand,
                self.eps,
                1.0 - self.eps,
                epsabs=self.epsabs,
                epsrel=self.epsrel,
                limit=200
            )
            return float(val_p ** (1.0 / p))

        else:
            raise ValueError("Unknown gamma_method.")
