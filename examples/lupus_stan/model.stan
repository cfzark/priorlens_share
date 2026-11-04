data {
  int<lower=1> N;
  vector[N] x;
  array[N] int<lower=0, upper=1> y;
  vector[2] prior_loc;
  vector<lower=0>[2] prior_scale;
  real<lower=0> prior_df;
}
parameters {
  real beta0;
  real beta1;
}
model {
  beta0 ~ student_t(prior_df, prior_loc[1], prior_scale[1]);
  beta1 ~ student_t(prior_df, prior_loc[2], prior_scale[2]);
  y ~ bernoulli_logit(beta0 + beta1 * x);
}
