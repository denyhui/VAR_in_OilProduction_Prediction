import numpy as np
import pandas as pd
from functools import reduce
import tkinter
import scipy.stats as stats
from tkinter import messagebox
from tkinter import filedialog
from statsmodels.compat.python import StringIO, lrange
from statsmodels.iolib import SimpleTable


def input_check(x):
    assert isinstance(x, (np.ndarray, pd.DataFrame, list)), 'input type must be in [ndarray, DataFrame, list]'
    if isinstance(x, list):
        x = np.array(x)
        assert x.dtype == np.float or x.dtype == np.int, 'input must be able to transform into num ndarray'
        assert len(x.shape) > 1 and x.shape[1] > 1, 'input must have multi variables'
    elif isinstance(x, np.ndarray):
        assert x.dtype == np.float or x.dtype == np.int, 'input dtype must be num'
        assert len(x.shape) > 1 and x.shape[1] > 1, 'input must have multi variables'
    else:
        assert all(i == np.float or i == np.int for i in x.dtypes), 'input dtype must be num'
        assert len(x.shape) > 1 and x.shape[1] > 1, 'input must have multi variables'


def num_check(x):
    assert isinstance(x, (float, int)), 'input must be num'


def chain_dot(*arr):
    return reduce(lambda x, y: np.dot(y, x), arr[::-1])


def _read():
    '''
    读取逻辑及异常逻辑
    :return: 读取结果
    '''
    file_path = filedialog.askopenfilename()
    if file_path is '':
        tkinter.messagebox.showerror('错误','未读取文件')
        raise RuntimeError
    try:
        suffix = file_path.split('.')[-1]
    except:
        tkinter.messagebox.showerror('错误', '读取失败')
        raise RuntimeError
    if suffix == 'csv':
        try:
            data = pd.read_csv(file_path)
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(file_path, encoding='gbk')
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(file_path, encoding='gb2312')
                except UnicodeDecodeError:
                    try:
                        data = pd.read_csv(file_path, encoding='gb18030')
                    except:
                        tkinter.messagebox.showerror('错误','编码错误')
                        raise RuntimeError
        except OSError:
            try:
                data = pd.read_csv(file_path, encoding='gbk', engine='python')
            except OSError:
                tkinter.messagebox.showerror('错误', '读取了错误的文件')
                raise OSError
    elif suffix == 'xlsx':
        try:
            data = pd.read_excel(file_path)
        except UnicodeDecodeError:
            try:
                data = pd.read_excel(file_path, encoding='gbk')
            except UnicodeDecodeError:
                try:
                    data = pd.read_excel(file_path, encoding='gb2312')
                except UnicodeDecodeError:
                    try:
                        data = pd.read_excel(file_path, encoding='gb18030')
                    except:
                        tkinter.messagebox.showerror('错误', '编码错误')
                        raise RuntimeError
        except OSError:
            try:
                data = pd.read_excel(file_path, encoding='gbk', engine='python')
            except OSError:
                tkinter.messagebox.showerror('错误', '读取了错误的文件')
                raise OSError
    else:
        tkinter.messagebox.showerror('错误', '未知错误')
        raise RuntimeError
    return data


def m_assert_error(flag, name):
    '''
    断言/错误函数
    :param flag: 断言内容
    :param name: 错误显示内容
    :return:
    '''
    if isinstance(flag, (tuple, list)):
        for f, n in zip(flag, name):
            try:
                assert f
            except:
                tkinter.messagebox.showerror('错误', n)
                raise RuntimeError
    else:
        try:
            assert flag
        except:
            tkinter.messagebox.showerror('错误', name)
            raise RuntimeError


def ma_rep(coefs, maxn=10):
    """
    Parameters
    ----------
    coefs : ndarray (p x k x k)
    maxn : int
        Number of MA matrices to compute
    Returns
    -------
    phis : ndarray (maxn + 1 x k x k)
    """
    p, k, k = coefs.shape
    phis = np.zeros((maxn + 1, k, k))
    phis[0] = np.eye(k)

    # recursively compute Phi matrices
    for i in range(1, maxn + 1):
        for j in range(1, i + 1):
            if j > p:
                break

            phis[i] += np.dot(phis[i - j], coefs[j - 1])

    return phis


def forecast_cov(ma_coefs, sigma_u, steps):
    """
    Parameters
    ----------
    steps : int
        Number of steps ahead

    Returns
    -------
    forc_covs : ndarray (steps x neqs x neqs)
    """
    neqs = len(sigma_u)
    forc_covs = np.zeros((steps, neqs, neqs))

    prior = np.zeros((neqs, neqs))
    for h in range(steps):
        # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
        phi = ma_coefs[h]
        var = chain_dot(phi, sigma_u, phi.T)
        forc_covs[h] = prior = prior + var

    return forc_covs


def forecast(y, coefs, trend_coefs, steps, exog=None):
    """
    Produce linear minimum MSE forecast

    Parameters
    ----------
    y : ndarray (k_ar x neqs)
    coefs : ndarray (k_ar x neqs x neqs)
    trend_coefs : ndarray (1 x neqs) or (neqs)
    steps : int
    exog : ndarray (trend_coefs.shape[1] x neqs)

    Returns
    -------
    forecasts : ndarray (steps x neqs)
    """
    p = len(coefs)
    k = len(coefs[0])
    # initial value
    forcs = np.zeros((steps, k))
    if exog is not None and trend_coefs is not None:
        forcs += np.dot(exog, trend_coefs)
    # to make existing code (with trend_coefs=intercept and without exog) work:
    elif exog is None and trend_coefs is not None:
        forcs += trend_coefs

    # h=0 forecast should be latest observation
    # forcs[0] = y[-1]

    # make indices easier to think about
    for h in range(1, steps + 1):
        # y_t(h) = intercept + sum_1^p A_i y_t_(h-i)
        f = forcs[h - 1]
        for i in range(1, p + 1):
            # slightly hackish
            if h - i <= 0:
                # e.g. when h=1, h-1 = 0, which is y[-1]
                prior_y = y[h - i - 1]
            else:
                # e.g. when h=2, h-1=1, which is forcs[0]
                prior_y = forcs[h - i - 1]

            # i=1 is coefs[0]
            f = f + np.dot(coefs[i - 1], prior_y)

        forcs[h - 1] = f

    return forcs


def _forecast_vars(steps, ma_coefs, sig_u):
    """
    Parameters
    ----------
    steps
    ma_coefs
    sig_u

    Returns
    -------

    """
    covs = forecast_cov(ma_coefs, sig_u, steps)
    # Take diagonal for each cov
    neqs = len(sig_u)
    inds = np.arange(neqs)
    return covs[:, inds, inds]


def forecast_interval(y, coefs, trend_coefs, sig_u, steps=5, alpha=0.05,
                      exog=1):
    assert (0 < alpha < 1)
    q = norm_signif_level(alpha)

    point_forecast = forecast(y, coefs, trend_coefs, steps, exog)
    ma_coefs = ma_rep(coefs, steps)
    sigma = np.sqrt(_forecast_vars(steps, ma_coefs, sig_u))

    forc_lower = point_forecast - q * sigma
    forc_upper = point_forecast + q * sigma

    return point_forecast, forc_lower, forc_upper


def log1p(x):
    return np.log1p(x)


def re_log1p(x):
    return np.exp(x) - 1


def norm_signif_level(alpha=0.05):
    return stats.norm.ppf(1 - alpha / 2)


def get_ic_table(ics, selected_orders):
    '''
    该方法将滞后阶数结果转换为表格化的分析结果
    :param ics: 滞后阶数结果
    :param selected_orders: 最大滞后阶数
    :return: 返回表格化的滞后阶数分析结果
    '''
    _default_table_fmt = dict(
        empty_cell='',
        colsep='  ',
        row_pre='',
        row_post='',
        table_dec_above='=',
        table_dec_below='=',
        header_dec_below='-',
        header_fmt='%s',
        stub_fmt='%s',
        title_align='c',
        header_align='r',
        data_aligns='r',
        stubs_align='l',
        fmt='txt'
    )
    cols = sorted(ics)
    data = np.array([["%#10.4g" % v for v in ics[c]] for c in cols],
                    dtype=object).T
    for i, col in enumerate(cols):
        idx = int(selected_orders[col]), i
        data[idx] = data[idx] + '*'
    fmt = dict(_default_table_fmt,
               data_fmts=("%s",) * len(cols))
    buf = StringIO()
    table = SimpleTable(data, cols, lrange(len(data)),
                        title='VAR Order Selection', txt_fmt=fmt)
    buf.write(str(table) + '\n')
    buf.write('* Minimum' + '\n')
    return buf.getvalue()


def mean_error_ratio(y_pred, y_true, axis=-1):
    return np.median(np.abs(y_pred - y_true) / (np.abs(y_true)), axis=axis)


def mae(y_pred, y_true, axis=-1):
    return np.mean(np.abs(y_true - y_pred), axis=axis)


def get_score(y_pred, y_true):
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mask = y_true > 1e-2
    y_pred, y_true = y_pred[mask], y_true[mask]
    score_mae = mae(y_pred=y_pred, y_true=y_true)
    score_mr = mean_error_ratio(y_pred=y_pred, y_true=y_true)
    return score_mae, score_mr


def predict_fix(y_pred):
    y_pred = re_log1p(y_pred)
    y_pred[np.isnan(y_pred)] = 0
    y_pred[np.isinf(y_pred)] = 0
    return y_pred
