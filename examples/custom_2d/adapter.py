"""Two-coefficient Gaussian regression: no optimized posterior is registered.

IS/NUTS use the generic interfaces; the analytic oracle is in tests only.
Input x is already processed. Its transformation is never re-estimated.
"""
import numpy as np
from scipy.stats import norm
from prioreffect_v2.core import DefinitionError


class UserRegression:
    name = 'user_regression'
    variables = ['beta0','beta1']
    dimension = 2
    unit = 'observation'
    stan_parameters = ['beta']

    def __init__(self,sigma=1.):
        self.sigma=float(sigma)
        if not np.isfinite(self.sigma) or self.sigma<=0:
            raise DefinitionError('sigma must be positive and finite')

    def validate_data(self,data):
        x,y=np.asarray(data['x'],float),np.asarray(data['y'],float)
        ids=list(data.get('unit_ids',[f'row_{i}' for i in range(len(y))]))
        if x.ndim!=1 or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or len(ids)!=len(y) or len(set(ids))!=len(ids):
            raise DefinitionError('Finite aligned x/y with unique row IDs required')
        return dict(x=x.tolist(),y=y.tolist(),unit_ids=ids)

    def validate_prior(self,prior):
        loc,sd=np.asarray(prior['loc']),np.asarray(prior['sd'])
        if prior.get('family')!='normal' or loc.shape!=(2,) or sd.shape!=(2,) or not np.isfinite(loc).all() or not np.isfinite(sd).all() or (sd<=0).any():
            raise DefinitionError('Two independent proper normal priors required')

    def check_posterior_admissibility(self,data,prior):
        self.validate_prior(prior)

    def check_target_moments(self,data,prior,target,order):
        self.validate_prior(prior)
        if order not in (1,2) or any(c.variable not in self.variables or c.transform!='identity' for c in target.coordinates):
            raise DefinitionError('Only affine coefficient moments certified')
        return dict(certified=True,order=order,rule='proper_Gaussian_posterior')

    def sampling_signature(self):
        return dict(sigma=self.sigma,design='empirical_rows')

    def subset(self,data,indices,auxiliary=False):
        ix=np.asarray(indices,int)
        return dict(x=np.asarray(data['x'])[ix].tolist(),y=np.asarray(data['y'])[ix].tolist(),
                    unit_ids=[f'aux_{i}' for i in range(len(ix))] if auxiliary else np.asarray(data['unit_ids'])[ix].tolist())

    def append(self,data,additions):
        return dict(x=data['x']+additions['x'],y=data['y']+additions['y'],
                    unit_ids=data['unit_ids']+[f'new_{len(data["y"])+i}' for i in range(len(additions['y']))])

    def simulate(self,theta,count,rng,design):
        x=np.asarray(design['x'])[rng.integers(len(design['x']),size=count)]
        y=rng.normal(theta[0]+theta[1]*x,self.sigma)
        return dict(x=x.tolist(),y=y.tolist(),unit_ids=[f'aux_{i}' for i in range(count)])

    def to_unconstrained(self,theta): return np.asarray(theta,float)
    def from_unconstrained(self,z): return np.asarray(z,float)

    def loglik(self,theta,data):
        b=np.atleast_2d(theta)
        mu=b[:,:1]+b[:,1:2]*np.asarray(data['x'])[None,:]
        return norm.logpdf(np.asarray(data['y'])[None,:],mu,self.sigma).sum(axis=1)

    def logprior(self,theta,prior):
        return norm.logpdf(np.atleast_2d(theta),prior['loc'],prior['sd']).sum(axis=1)

    def logkernel(self,z,data,prior):
        return self.loglik(z,data)+self.logprior(z,prior)

    def stan_data(self,data,prior):
        return dict(N=len(data['y']),x=data['x'],y=data['y'],sigma=self.sigma,
                    prior_loc=prior['loc'],prior_sd=prior['sd'])

    def stan_draws(self,fit): return fit.stan_variable('beta')


def create_model(registration):
    return UserRegression(**registration.get('options',{}))
