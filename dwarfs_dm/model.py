
import dynesty
import numpy as np

class Model():
    ''' Partent class to fit a model with log likelihood using Nested Sampling '''
    def __init__(self, params_list):
        '''
        - priors: (dict) dictionary with prior range
        '''
        self.results = None
        self.weights = None
        self.params_list = params_list
        self.priors = {}
        self.n_params = len(self.params_list)

    def log_likelihood(self, x):
        ''' Log likelihood '''
        pass

    def priors_fn(self, u):
        '''Transforms samples `u` drawn from the unit to our prior'''
        for i, params in enumerate(self.params_list):
            u_min = self.priors[params][0]
            u_max = self.priors[params][1]
            u[i] = (u_max - u_min) * u[i] + u_min
        return u

    def sample(self, *args, **kargs):
        ''' Nested sampling with dynesty '''
        sampler = dynesty.NestedSampler(
            self.log_likelihood, self.priors_fn, self.n_params,
            *args, **kargs)
        sampler.run_nested()
        self.results = sampler.results
        self.weights = np.exp(self.results.logwt - self.results.logz[-1])
        #self._write_results(outfile)

    def get_median(self, key=None):
        ''' Get the best parameters '''

        if key is not None:
            index = self.params_list.index(key)
            return dynesty.utils.quantile(
                self.results.samples[:, index], [0.5, ], weights=self.weights)[0]

        median = {}
        for i, params in enumerate(self.params_list):
            median[params] = dynesty.utils.quantile(
                self.results.samples[:, i], [0.5, ], weights=self.weights)[0]
        return median


    #def _write_results(self, outfile):
    #    if outfile is not None:
    #        with open(outfile, 'wb') as f:
    #            pickle.dump(self.results, f)


