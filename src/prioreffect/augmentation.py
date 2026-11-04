import numpy as np

# Augmentation strategies
class Augmenter:
    def generate(self, y, model, prior_inf, posterior_inf, M, rng=None):
        raise NotImplementedError


class BootstrapAugmenter(Augmenter):
    def generate(self, y, model, prior_inf, posterior_inf, M, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        y = np.asarray(y, dtype=float)
        return rng.choice(y, size=M, replace=True)


class PosteriorPredictiveAugmenter(Augmenter):
    def generate(self, y, model, prior_inf, posterior_inf, M, rng=None):
        return model.posterior_predictive(posterior_inf, size=M, rng=rng)


class PriorPredictiveAugmenter(Augmenter):
    def generate(self, y, model, prior_inf, posterior_inf, M, rng=None):
        return model.prior_predictive(prior_inf, size=M, rng=rng)