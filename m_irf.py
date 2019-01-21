import numpy as np
import numpy.linalg as la
from statsmodels.compat.python import lrange
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tools.tools import chain_dot

class m_irf:
    def __init__(self, model, periods=10, P=None, order=None, svar=False):
        self.model = model
        self.periods = periods
        self.neqs, self.lags, self.T = model.neqs, model.k_ar, model.nobs

        self.order = order

        if P is None:
            sigma = model.sigma_u

            P = la.cholesky(sigma)

        self.P = P

        self.svar = svar

        self.irfs = model.ma_rep(periods)
        if svar:
            self.svar_irfs = model.svar_ma_rep(periods, P=P)
        else:
            self.orth_irfs = model.orth_ma_rep(periods, P=P)

        self.cum_effects = self.irfs.cumsum(axis=0)
        if svar:
            self.svar_cum_effects = self.svar_irfs.cumsum(axis=0)
        else:
            self.orth_cum_effects = self.orth_irfs.cumsum(axis=0)

        self.lr_effects = model.long_run_effects()
        if svar:
            self.svar_lr_effects = np.dot(model.long_run_effects(), P)
        else:
            self.orth_lr_effects = np.dot(model.long_run_effects(), P)


        # auxiliary stuff
        self._A = util.comp_matrix(model.coefs)

    def cov(self, orth=False):
        if orth:
            return self._orth_cov()

        covs = self._empty_covm(self.periods + 1)
        covs[0] = np.zeros((self.neqs ** 2, self.neqs ** 2))
        for i in range(1, self.periods + 1):
            Gi = self.G[i - 1]
            covs[i] = chain_dot(Gi, self.cov_a, Gi.T)

        return covs

    def plot(self, orth=False, impulse=None, response=None,
             signif=0.05, plot_params=None, subplot_params=None,
             repl=1000,
             seed=None, component=None):

        periods = self.periods
        model = self.model
        svar = self.svar

        if orth and svar:
            raise ValueError("For SVAR system, set orth=False")

        if orth:
            title = 'Impulse responses (orthogonalized)'
            irfs = self.orth_irfs
        elif svar:
            title = 'Impulse responses (structural)'
            irfs = self.svar_irfs
        else:
            title = 'Impulse responses'
            irfs = self.irfs

        stderr = None
        stderr_type='asym'


        plotting.irf_grid_plot(irfs, stderr, impulse, response,
                               self.model.names, title, signif=signif,
                               subplot_params=subplot_params,
                               plot_params=plot_params, stderr_type=stderr_type)



