data {
  int<lower=0> N;
  vector[N] y;
  real<lower=0> sigma;
  real prior_mean;
  real<lower=0> prior_sd;
}
parameters { real mu; }
model {
  mu ~ normal(prior_mean, prior_sd);
  y ~ normal(mu, sigma);
}
