import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.tsa.vector_ar.util as util

import tkinter
from tkinter import ttk
import tkinter.messagebox

from util import m_assert_error, mean_error_ratio, mae
from functools import partial
import re
import tqdm


class m_irf_window:
    '''
    脉冲响应窗口
    '''

    def __init__(self, model, total, p_col_num, i_col_num):
        '''
        脉冲响应窗口初始化函数
        :param model: VAR模型
        :param results: VAR拟合结果
        :param total: 表格化数据体
        :param p_col_num: 采出井索引
        :param i_col_num: 注入井索引
        '''
        self.model = model
        self.total = total
        self.z = self.model.z
        self.sigma_u = self.model.sigma_u
        self.p_col_num = p_col_num
        self.i_col_num = i_col_num
        self.periods = 10
        self.current_inj = []

        self.root = tkinter.Tk()
        self.root.title('脉冲响应分析')
        self.root.geometry('250x220')

        tkinter.Label(self.root, text='采出井号').grid(row=0)
        self.cl_1 = ttk.Combobox(self.root, )
        self.cl_1.grid(row=1, columnspan=3)

        self.cl_1['values'] = ['所有'] + ['present_' + c if self.total[c].iloc[-1] > 0.01 else c for c in
                                        self.total.columns[self.p_col_num]]

        self.cl_1.current(0)
        self.cl_1['state'] = 'readonly'
        for i, c in enumerate(self.total.columns[self.p_col_num]):
            if self.total[c].iloc[-1] > 0.01:
                self.cl_1.rowconfigure(0, {'weight': 3})

        tkinter.Label(self.root, text='注入井号').grid(row=2)
        self.cl_2 = ttk.Combobox(self.root, )
        self.cl_2.grid(row=3, columnspan=3)
        self.cl_2['values'] = ['所有'] + ['present_' + c if self.total[c].iloc[-1] > 0.01 else c for c in
                                        self.total.columns[self.i_col_num]]
        self.cl_2.current(0)
        self.cl_2['state'] = 'readonly'
        for i, c in enumerate(self.total.columns[self.i_col_num]):
            if self.total[c].iloc[-1] > 0.01:
                self.cl_2.rowconfigure(0, {'weight': 3})
                self.current_inj.append((i, c))

        self.max_n = 10

        #self.phis = self.exog_irf(self.model, self.max_n)
        self.phis = model.exog_irf(self.max_n)
        self.stderr = None

        tkinter.Button(self.root, text='影响图',
                       command=self.plot).grid(row=4, column=0, sticky=tkinter.W)
        tkinter.Button(self.root, text='累积影响图',
                       command=partial(self.plot, cumplot=True)).grid(row=4, column=1, sticky=tkinter.W)
        tkinter.Button(self.root, text='分析',
                       command=self.analysis).grid(row=4, column=2, sticky=tkinter.W)

        self.v = tkinter.IntVar()
        self.v.set(0)
        tkinter.Checkbutton(self.root, text='计算不确定性', variable=self.v, ).grid(row=5, sticky=tkinter.W, columnspan=3)
        self.canvas = tkinter.Canvas(self.root, width=170, height=26, bg="white")
        self.canvas.grid(row=6, columnspan=3)
        self.fill_line = self.canvas.create_rectangle(2, 2, 0, 27, width=0, fill="blue")
        self.process_vs = tkinter.StringVar()
        self.process_vs.set('')
        self.process_vl = tkinter.Label(self.root, fg='blue', textvariable=self.process_vs)
        self.process_vl.grid(row=7, columnspan=3)

        self.root.mainloop()

    def process_bar(self, x, i):
        '''
        进度条
        :param x: 最大循环次数
        :param i: 当前循环次数
        :return:
        '''
        n = i * 180 / x
        self.canvas.coords(self.fill_line, (0, 0, n, 30))
        self.process_vs.set(str(round(i + 1 / x, 1)) + "%")
        self.root.update()

    def plot(self, cumplot=False):
        '''
        绘制脉冲响应图,目前的不确定性分析尚未完成
        :return:
        '''
        resp_name = self.cl_1.get()
        imp_name = self.cl_2.get()

        if resp_name == '所有' and imp_name == '所有':
            tkinter.messagebox.showerror('错误', '不能同时选择所有注入井及采出井')
            raise RuntimeError
        elif resp_name == '所有':
            j = self.cl_2['values'].index(imp_name) - 1
            fig, ax = plt.subplots()
            if cumplot:
                plt.plot(np.arange(self.phis.shape[0] + 1),
                         np.concatenate([np.array([0]), self.phis[:, j + 1, :].sum(axis=1).cumsum()]))
                plt.ylabel('累积影响', fontsize=15)
            else:
                plt.plot(np.arange(self.phis.shape[0]) + 1, self.phis[:, j + 1, :].sum(axis=1))
                plt.ylabel('影响', fontsize=15)
            plt.title(r'%s %s$\rightarrow$%s' %
                      ('累积影响' if cumplot else '',
                       re.sub('present_', '', imp_name),
                       '所有'),
                      fontsize=20)
            plt.xlabel('步长', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
        elif imp_name == '所有':
            i = self.cl_1['values'].index(resp_name) - 1
            fig, ax = plt.subplots()
            if cumplot:
                plt.plot(np.arange(self.phis.shape[0] + 1),
                         np.concatenate([np.array([0]), self.phis[:, :, i].sum(axis=1).cumsum()]))
                plt.ylabel('累积影响', fontsize=15)
            else:
                plt.plot(np.arange(self.phis.shape[0]) + 1, self.phis[:, :, i].sum(axis=1))
                plt.ylabel('影响', fontsize=15)
            plt.title(r'%s %s$\rightarrow$%s' %
                      ('累积影响' if cumplot else '',
                       '所有',
                       re.sub('present_', '', resp_name)),
                      fontsize=20)
            plt.xlabel('步长', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
        else:
            i = self.cl_1['values'].index(resp_name)
            j = self.cl_2['values'].index(imp_name)
            fig, ax = plt.subplots()
            if cumplot:
                plt.plot(np.arange(self.phis.shape[0] + 1),
                         np.concatenate([np.array([0]), self.phis[:, j + 1, i].cumsum()]))
                plt.ylabel('累积影响', fontsize=15)
            else:
                plt.plot(np.arange(self.phis.shape[0]) + 1, self.phis[:, j + 1, i])
                plt.ylabel('影响', fontsize=15)
            plt.title(r'%s %s$\rightarrow$%s' %
                      ('累积影响' if cumplot else '',
                       re.sub('present_', '', resp_name),
                       re.sub('present_', '', imp_name)),
                      fontsize=20)
            plt.xlabel('步长', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            if self.v.get() == 1:
                if self.stderr is None:
                    self.stderr = self._compute_std()
                plt.fill_between(np.arange(self.phis.shape[0]),
                                 self.phis[:, j, i] + self.stderr[0][:, j, i],
                                 self.phis[:, j, i] - self.stderr[1][:, j, i], alpha=0.2)
        plt.show()

    def _compute_std(self, repl=100, signif=0.05, burn=100, cum=False, ):
        '''
        不确定性分析函数
        :param repl:
        :param signif:
        :param burn:
        :param cum:
        :return:
        '''
        ex_neqs = self.model.trend_coefs.shape[1]
        neqs = self.model.neqs
        k_ar = self.model.k_ar
        coefs = self.model.coefs
        sigma_u = self.sigma_u
        intercept = self.results.intercept
        nobs = self.results.nobs

        ma_coll = np.zeros((repl, self.max_n + 1, neqs, ex_neqs))

        def fill_coll(sim):
            ret = VAR(sim, exog=self.model.exog[-nobs:]).fit(maxlags=k_ar, )
            ret = self.exog_irf(self.results, self.max_n)
            return ret.cumsum(axis=0) if cum else ret

        for i in tqdm.tqdm(range(repl)):
            sim = util.varsim(coefs, intercept, sigma_u,
                              seed=None, steps=nobs + burn)
            sim = sim[burn:]
            ma_coll[i, :, :, :] = fill_coll(sim)
            self.process_bar(repl, i)

        ma_sort = np.sort(ma_coll, axis=0)  # sort to get quantiles
        low_idx = int(round(signif / 2 * repl) - 1)
        upp_idx = int(round((1 - signif / 2) * repl) - 1)
        lower = ma_sort[low_idx, :, :, :]
        upper = ma_sort[upp_idx, :, :, :]
        return lower, upper

    def analysis(self):
        '''
        对模型进行脉冲响应分析
        :return:
        '''
        n = len(self.current_inj)
        i, c = list(zip(*self.current_inj))
        c = [re.sub('_inj', '', c_) for c_ in c]
        v = self.phis[:, np.array(i) + 1, :].sum(axis=2).sum(axis=0)
        DF_data = pd.DataFrame(v, columns=['累积影响'])
        DF_data['井号'] = c

        # sns.barplot(data=DF_data, x='井号', y='累积影响', color="b")
        plt.stem(np.arange(len(v)), v, linefmt='b-', markerfmt='o', linewidth=2, markersize=2)
        plt.axhline(0, ls=":", c=".5")
        plt.xlabel('井号', fontsize=10)
        plt.ylabel('累积影响', fontsize=15)
        plt.xticks(np.arange(len(v)), c, fontsize=15, rotation=25)
        plt.yticks(fontsize=15)
        plt.grid()
        plt.show()


class m_forecast_plot_window:
    '''
    预测图绘制窗口
    '''

    def __init__(self, Y_endog, Y_future, Y_pred, fore_cov=None):
        '''
        预测图绘制窗口初始化函数
        :param Y_endog: 训练用数据
        :param Y_future: 待预测数据
        :param Y_pred: 预测结果
        :param result: 拟合结果数据体
        '''
        self.Y_endog = Y_endog
        self.Y_future = Y_future
        self.Y_pred = Y_pred
        self.fore_cov = fore_cov
        self.root = tkinter.Tk()
        self.root.title('预测图形绘制')
        self.root.geometry('225x300')
        self.prod_lb = tkinter.Listbox(self.root, selectmode=tkinter.MULTIPLE)
        self.current_prod = []
        for c in Y_endog.columns:
            self.prod_lb.insert(tkinter.END, c)
        for i, c in enumerate(Y_endog.columns):
            if Y_endog[c].iloc[-1] < 1e-2:
                self.prod_lb.itemconfig(i, {'fg': 'grey'})
            else:
                self.current_prod.append((i, c))
        self.prod_lb.grid()
        self.prod_lb_scroll = tkinter.Scrollbar(self.root, command=self.prod_lb.yview)
        self.prod_lb_scroll.grid(row=0, column=1, sticky='ns')
        self.prod_lb.config(yscrollcommand=self.prod_lb_scroll.set)
        self.prod_lb.selection_set(0)

        tkinter.Label(self.root, text='灰色为近期未投产的井', fg='blue').grid(row=1, column=0, columnspan=2)
        tkinter.Button(self.root, text='作图', command=self.plot).grid(row=2, column=0)
        tkinter.Button(self.root, text='分析', command=self.analysis).grid(row=2, column=1)

        # forecast_interval(Y_endog, self.result.coefs,self.result.exog_coefs, self.result.sigma_u, steps=)
        self.root.mainloop()

    def plot(self):
        '''
        绘图函数
        :return:
        '''
        sl = self.prod_lb.curselection()
        m_assert_error(len(sl) > 0, '必须选中至少一口井')
        w_name = [self.prod_lb.get(i) for i in sl] if len(sl) > 1 else [self.prod_lb.get(sl)]
        w_num = [list(self.Y_endog.columns).index(c) for c in w_name]

        c_b = ['r', 'b', 'c', 'g', 'y', 'k', 'm']
        fig, ax = plt.subplots()
        i_ = 0
        for i_, i in enumerate(w_num):
            c = w_name[i_]
            plt.plot(self.Y_endog.index, self.Y_endog[c], c_b[i_ % 7] + '.', label='%s原始' % c, markersize=2)
            plt.plot(self.Y_future.index, self.Y_pred[:, i], c_b[i_ % 7] + '-d', label='%s预测' % c, linewidth=3)
            plt.plot(self.Y_future.index, self.Y_future[c], c_b[i_ % 7] + 'x', label='%s实际' % c)
            if self.fore_cov is not None:
                forc_lower, forc_upper = self.fore_cov
                plt.fill_between(self.Y_future.index, forc_lower[:, i], forc_upper[:, i], label='%s不确定范围' % c)
        plt.legend(loc='best', fontsize=15)
        plt.xlabel('时间', fontsize=15)
        plt.ylabel('日采出量/(m^3/d)', fontsize=15)
        plt.xticks(fontsize=15, rotation=75)
        plt.yticks(fontsize=15)
        plt.show()

    def analysis(self):
        '''
        分析预测误差分布
        :return:
        '''
        n = len(self.current_prod)
        Y_pred, Y_future = np.array(self.Y_pred), np.array(self.Y_future)
        i, c = list(zip(*self.current_prod))
        _y_pred = Y_pred[:, i]
        _y_future = Y_future[:, i]
        _y_pred_mean = np.mean(_y_pred, axis=0)
        _y_future_mean = np.mean(_y_future, axis=0)
        _mae = mae(y_pred=_y_pred, y_true=_y_future, axis=0)
        _mer = mean_error_ratio(y_pred=_y_pred, y_true=_y_future, axis=0)
        _y_pred_DF, _y_future_DF = pd.DataFrame(_y_pred_mean, columns=['数值']), pd.DataFrame(_y_future_mean,
                                                                                            columns=['数值'])
        _y_pred_DF['标签'], _y_future_DF['标签'] = ['预测'] * n, ['实际'] * n
        _y_pred_DF['井号'] = _y_future_DF['井号'] = c
        _y_pred_DF['误差'] = _y_future_DF['误差'] = _mer
        _y_DF = pd.concat([_y_pred_DF, _y_future_DF], axis=0)
        plt.subplot(2, 1, 1)
        sns.barplot(data=_y_DF, x='井号', y='数值', hue='标签', )
        plt.xticks([])
        plt.yticks(fontsize=15)
        plt.subplot(2, 1, 2)
        sns.pointplot(data=_y_DF, x='井号', y='误差', )
        plt.ylim(0, 1)
        plt.xticks(fontsize=15, rotation=15)
        plt.yticks(fontsize=15)
        plt.grid()
        plt.show()
