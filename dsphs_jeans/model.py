
import numpy as np
import bilby


class Model(bilby.Likelihood):
    ''' Log likelhood model template '''
    def __init__(self, parameters):
        super().__init__(parameters=parameters)
        self.result = None
        self.priors = {}

    def log_likelihood(self):
        ''' Log likelihood '''
        pass

    def run_sampler(self, *args, **kargs):
        ''' Run sampler '''
        self.result = bilby.run_sampler(
            likelihood=self, priors=self.priors, *args, **kargs)


    def get_credible_intervals(self, key, p=0.95):
        ''' Return the credible intervals of the posterior of a given key '''
        lo = (1 - p) / 2 * 100
        hi = (1 + p) / 2 * 100
        values = self.result.posterior[key].values
        return np.percentile(values, q=[lo, hi])

    def get_median(self, key):
        ''' Return the median of the posterior of a given key '''
        values = self.result.posterior[key].values
        return np.percentile(values, q=50)

    def get_mean_and_std(self, key):
        ''' Return the mean and standard deviation of the posterior of a given key '''
        values = self.result.posterior[key].values
        return np.mean(values), np.std(values)

