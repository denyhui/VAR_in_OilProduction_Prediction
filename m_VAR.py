'''
重新造的轮子
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from util import input_check, num_check, m_assert_error, forecast, forecast_interval


class m_VAR_model:
    def __init__(self, endog, exog=None, maxlags=1, l2=1):
        input_check(endog)
        if exog is not None:
            input_check(exog)
            assert len(endog) == len(exog), 'T of endog and exog must be equal'
        if maxlags != 1:
            num_check(maxlags)
            assert maxlags < len(endog), 'maxlags must be smaller than T'
        self.endog = endog
        self.exog = exog
        self.maxlags = maxlags
        self.coefs = None
        self.trend_coefs = None
        self.sigma_u = None
        self.l2 = l2
        self.p_exog = 2

    def fit(self, ):
        '''
        y:K * (T-p)
        Z:(1+K*p) * (T-p)
        B_est:K * (1+K*p)
        :return:
        '''
        Y = self.endog
        K = self.endog.shape[1]
        T, p = len(Y), self.maxlags
        Y_exog = self.exog
        self.p_exog = min(self.p_exog, p)
        y_sample = Y[p:]

        z = np.ones((T - p, 1))
        if Y_exog is not None:
            z = np.column_stack([z, *[Y_exog[p - i:T - i] for i in range(self.p_exog)]])
        z = np.concatenate([z, *[Y[p - i:T - i] for i in range(1, p + 1)]], axis=1)
        lr = Ridge(fit_intercept=False, alpha=self.l2).fit(z, y_sample)
        params = lr.coef_.T
        resid = y_sample - np.dot(z, params)

        self.params = params.astype(np.float32)
        df_resid = T - K * p - 1
        df_resid = df_resid if df_resid > 0 else 10
        mse = np.dot(resid.T, resid)
        self.sigma_u = mse / df_resid
        self.z = z
        if Y_exog is not None:
            self.trend_coefs = params[:(1 + self.p_exog * Y_exog.shape[1])]
            self.coefs = params[(1 + self.p_exog * Y_exog.shape[1]):].reshape((-1, K, K))
            self.coefs = np.concatenate([self.coefs[i].T[None, :, :] for i in range(self.coefs.shape[0])], axis=0)
        else:
            self.trend_coefs = params[:1]
            self.coefs = params[1:].reshape((-1, K, K))
            self.coefs = np.concatenate([self.coefs[i].T[None, :, :] for i in range(self.coefs.shape[0])], axis=0)

    def is_stable(self):
        pass

    def exog_irf(self,  max_n):
        '''
        计算脉冲响应期望均值函数
        :param results: VAR拟合结果
        :param max_n: 最大脉冲影响阶数
        :return: 脉冲影响矩阵
        '''
        p, k, k = self.coefs.shape
        k_i = self.exog.shape[1]
        p_i = int((self.trend_coefs.shape[0] - 1) / k_i)
        # k_i = self.trend_coefs.shape[0]
        phis = np.zeros((max_n + 1, 1 + k_i, k))
        param_c = self.trend_coefs[:1, :]
        param_i = self.trend_coefs[1:, :].reshape((-1, k_i, k))
        phis[0][1:] = np.dot(np.eye(k_i), param_i[0])
        for i in range(1, max_n + 1):
            for j in range(1, i + 1):
                #if j < -1:
                #    phis[i][1:] += np.dot(np.eye(k_i), param_i[j])
                if j > p:
                    break
                phis[i] += np.dot(phis[i - j], self.coefs[j - 1], )

        return phis

    def forecast_calibration(self, steps, exog_future=None):
        assert self.coefs is not None, 'must be fitted!'
        num_check(steps)
        if self.exog is None and exog_future is not None:
            raise RuntimeError
        if self.exog is not None and exog_future is None:
            raise RuntimeError
        if exog_future is not None:
            exog_future = np.array(exog_future)
            x_inst = np.row_stack(
                [np.column_stack(
                    [exog_future[h - i][None, :] if h - i >= 0 else np.array(self.exog)[h - i][None, :] for i in
                     range(self.p_exog)])
                    for h
                    in range(steps)])
            exog = np.column_stack([np.ones((steps, 1)), x_inst])
        else:
            exog = np.ones((steps, 1))
        return exog

    def forecast(self, Y_endog, steps, exog_future=None, ):
        exog = self.forecast_calibration(steps, exog_future=exog_future)
        Y_endog = np.array(Y_endog)
        forcs = forecast(Y_endog, self.coefs, self.trend_coefs, steps, exog=exog)
        return forcs

    def forecast_interval(self, Y_endog, steps, exog_future=None):
        exog = self.forecast_calibration(steps, exog_future=exog_future)
        Y_endog = np.array(Y_endog)
        point_forecast, forc_lower, forc_upper = forecast_interval(Y_endog, self.coefs,
                                                                   self.trend_coefs, self.sigma_u,
                                                                   steps=steps, exog=exog)
        return point_forecast, forc_lower, forc_upper

    def mse(self, coefs, trend_coefs, sigma_u, steps):
        from util import ma_rep
        ma_coefs = ma_rep(coefs, steps)
        pass


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from util import input_check
    from statsmodels.tsa.vector_ar.var_model import VAR

    t = np.arange(1, 100, ) / 10
    t = t.reshape((-1, 1))
    Y = np.concatenate([np.sin(t) * t, np.cos(t) * t, np.sin(t) + t, np.cos(t) - t], axis=1)
    var = m_VAR_model(Y[:-10], maxlags=5)
    var.fit()
    var = VAR(Y[:-10])
    res = var.fit(maxlags=5)
    res.plot_forecast(5)
