// Reference likelihoods evaluated from observation rows. Hyperparameter
// Eight Schools likelihood is collapsed over the unit effects analytically.
data {
  int<lower=1,upper=6> kind;
  int<lower=1,upper=2> D;
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
  vector<lower=0>[N] s;
  real<lower=0> sigma;
  int<lower=1> trials;
  int<lower=1,upper=3> family;
  real mu0;
  real<lower=0> prior_sd;
  real<lower=0> a;
  real<lower=0> b;
  real<lower=0> tau0_sq;
  int<lower=1,upper=4> tau_kind;
  real<lower=0> tau_scale;
  real<lower=0> tau_df;
  real<lower=0> tau_log_sd;
  vector[2] beta_loc;
  vector<lower=0>[2] beta_scale;
  real<lower=0> beta_df;
}
parameters { vector[D] z; }
transformed parameters {
  vector[D] theta = z;
  if (kind == 2 || kind == 4 || kind == 5) theta[D] = exp(z[D]);
  if (kind == 3) theta[1] = inv_logit(z[1]);
}
model {
  if (kind == 1) {
    if (family == 1) theta[1] ~ normal(mu0, prior_sd);
    y ~ normal(theta[1], sigma);
  } else if (kind == 2) {
    if (family == 1) theta[1] ~ gamma(a,b);
    if (family == 3) target += -0.5*z[1];
    target += z[1];
    for (i in 1:N) target += y[i]*z[1] - theta[1] - lgamma(y[i]+1);
  } else if (kind == 3) {
    if (family == 1) theta[1] ~ beta(a,b);
    if (family == 3) target += -0.5*(log(theta[1])+log1m(theta[1]));
    target += log_inv_logit(z[1]) + log1m_inv_logit(z[1]);
    for (i in 1:N) target += lgamma(trials+1)-lgamma(y[i]+1)-lgamma(trials-y[i]+1)
                            +y[i]*log_inv_logit(z[1])+(trials-y[i])*log1m_inv_logit(z[1]);
  } else if (kind == 4) {
    if (family == 1 || family == 2) theta[2] ~ inv_gamma(a,b);
    if (family == 1) theta[1] ~ normal(mu0,sqrt(tau0_sq*theta[2]));
    if (family == 3) target += -z[2];
    target += z[2];
    y ~ normal(theta[1],sqrt(theta[2]));
  } else if (kind == 5) {
    theta[1] ~ normal(mu0,prior_sd);
    if (tau_kind == 1) target += student_t_lpdf(theta[2]|tau_df,0,tau_scale)+log(2);
    if (tau_kind == 2) theta[2] ~ lognormal(log(tau_scale),tau_log_sd);
    if (tau_kind == 3) target += cauchy_lpdf(theta[2]|0,tau_scale)+log(2);
    if (tau_kind == 4) theta[2] ~ uniform(0,tau_scale);
    target += z[2];
    y ~ normal(theta[1],sqrt(square(s)+square(theta[2])));
  } else {
    theta ~ student_t(beta_df,beta_loc,beta_scale);
    for (i in 1:N) target += y[i]*(theta[1]+theta[2]*x[i])-log1p_exp(theta[1]+theta[2]*x[i]);
  }
}
