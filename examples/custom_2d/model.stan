data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
  real<lower=0> sigma;
  vector[2] prior_loc;
  vector<lower=0>[2] prior_sd;
}
parameters { vector[2] beta; }
model {
  beta ~ normal(prior_loc, prior_sd);
  y ~ normal(beta[1] + beta[2] * x, sigma);
}
