from __future__ import absolute_import
# GUI
import tkinter
from tkinter import ttk
import tkinter.messagebox
# basic
import numpy as np
import pandas as pd

import scipy
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.compat.python import iteritems

# plotting
import matplotlib.pyplot as plt

# util
import re
from auxiliary_window import m_forecast_plot_window, m_irf_window
from collections import defaultdict
from util import m_assert_error,get_ic_table
from util import get_score,predict_fix
from preprocess import table2timeseries

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class window:
    def __init__(self):
        np.random.seed(1)
        self.Pdata = None
        self.Idata = None
        self.total = None
        self.result = None
        self.model = None
        self.root = tkinter.Tk()
        self.root.title('自向量回归应用')
        self.root.geometry('480x490')

        self.tabcontrol = ttk.Notebook(self.root)
        self.tab1 = ttk.Frame(self.tabcontrol)
        self.tabcontrol.add(self.tab1, text='输入')
        self.tab2 = ttk.Frame(self.tabcontrol)
        self.tabcontrol.add(self.tab2, text='滞后阶数分析')
        self.tabcontrol.grid()

        self.read_frm = ttk.Frame(self.tab1)
        tkinter.Label(self.read_frm, text='输入参数', fg='blue').grid(row=0, column=0, sticky=tkinter.W)
        tkinter.Label(self.read_frm, text='表格内年月表头').grid(row=1, column=0)
        self.P_cl_1 = ttk.Combobox(self.read_frm)
        self.P_cl_1.grid(row=2, column=0)
        self.P_cl_1['state'] = 'readonly'
        tkinter.Label(self.read_frm, text='表格内井号表头').grid(row=3, column=0)
        self.P_cl_2 = ttk.Combobox(self.read_frm)
        self.P_cl_2.grid(row=4, column=0)
        self.P_cl_2['state'] = 'readonly'
        tkinter.Label(self.read_frm, text='表格内单位时间产量表头').grid(row=5, column=0)
        self.P_cl_3 = ttk.Combobox(self.read_frm)
        self.P_cl_3.grid(row=6, column=0)
        self.P_cl_3['state'] = 'readonly'

        tkinter.Button(self.read_frm, text='读取采出井数据', command=self.read_csv_P).grid(row=7, column=0)

        tkinter.Label(self.read_frm, text='表格内年月表头').grid(row=1, column=1)
        self.I_cl_1 = ttk.Combobox(self.read_frm)
        self.I_cl_1.grid(row=2, column=1)
        self.I_cl_1['state'] = 'readonly'
        tkinter.Label(self.read_frm, text='表格内井号表头').grid(row=3, column=1)
        self.I_cl_2 = ttk.Combobox(self.read_frm)
        self.I_cl_2.grid(row=4, column=1)
        self.I_cl_2['state'] = 'readonly'
        tkinter.Label(self.read_frm, text='表格内单位时间注入量表头').grid(row=5, column=1)
        self.I_cl_3 = ttk.Combobox(self.read_frm)
        self.I_cl_3.grid(row=6, column=1)
        self.I_cl_3['state'] = 'readonly'
        tkinter.Button(self.read_frm, text='读取注入井数据', command=self.read_csv_I).grid(row=7, column=1)

        self.aux_frm = tkinter.Frame(self.tab1)

        self.lag_command = tkinter.Button(self.aux_frm, text='滞后阶数分析', command=self.maxlag_selection, width=10,
                                          state=tkinter.DISABLED)
        self.lag_command.grid(row=4, column=0, sticky=tkinter.W)

        self.validate_command = tkinter.Button(self.aux_frm, text='拟合并且验证', command=self.fit_and_validate, width=10,
                                               state=tkinter.DISABLED)
        self.validate_command.grid(row=4, column=1, sticky=tkinter.W)

        self.predict_command = tkinter.Button(self.aux_frm, text='拟合', command=self.fit, width=10,
                                              state=tkinter.DISABLED)
        self.predict_command.grid(row=4, column=2, sticky=tkinter.W)

        tkinter.Label(self.aux_frm, text='信息', fg='blue').grid(row=6)
        self.verbose_list = tkinter.Listbox(self.aux_frm, width=55, height=4)
        self.verbose_list.grid(row=7, column=0, columnspan=4)
        self.ver_lb_scroll = tkinter.Scrollbar(self.aux_frm, command=self.verbose_list.yview)
        self.ver_lb_scroll.grid(row=7, column=4, sticky='ns')
        self.verbose_list.config(yscrollcommand=self.ver_lb_scroll.set)
        self.verbose_list.selection_set(0)

        tkinter.Label(self.aux_frm, text='滞后阶数', ).grid(row=1, column=0, columnspan=2, )
        self.Lag_cl = ttk.Combobox(self.aux_frm)
        self.Lag_cl.grid(row=2, column=0, columnspan=2)
        self.Lag_cl['values'] = list(range(1, 25))
        self.Lag_cl.current(0)
        self.Lag_cl['state'] = 'readonly'

        def new_root_permission():
            self.new_button['state'] = tkinter.NORMAL if self.v2.get() else tkinter.DISABLED

        self.v2 = tkinter.IntVar()
        self.v2.set(0)
        tkinter.Checkbutton(self.aux_frm,
                            text='使用注入井数据',
                            variable=self.v2,
                            command=new_root_permission).grid(row=3,column=0, columnspan=2)
        self.v3 = tkinter.IntVar()
        self.v3.set(0)
        tkinter.Checkbutton(self.aux_frm,
                            text='预测不确定性',
                            variable=self.v3, ).grid(row=3, column=2, columnspan=2)

        self.new_button = tkinter.Button(self.aux_frm, text='脉冲响应分析', command=self.m_irf, state=tkinter.DISABLED,
                                         width=10, )
        self.new_button.grid(row=4, column=3, sticky=tkinter.W)

        tkinter.Label(self.aux_frm, text='正则化系数', ).grid(row=1, column=2, columnspan=2, )
        self.re_cl = ttk.Combobox(self.aux_frm)

        self.re_cl['values'] = list(range(100))
        self.re_cl.current(2)
        self.re_cl['state'] = 'readonly'
        self.re_cl.grid(row=2, column=2, columnspan=2)

        self.Info_frm = tkinter.Frame(self.tab2)
        tkinter.Label(self.Info_frm, text='分析结果', fg='blue').grid(sticky=tkinter.W)
        self.InfoText = tkinter.Text(self.Info_frm, width=60, height=30)
        self.InfoText.grid()
        self.Info_frm.grid(row=0, column=1, rowspan=2)

        self.read_frm.grid(row=0, column=0,)
        self.aux_frm.grid( row=1, column=0,)

        def get_help():
            temp_info = '''
            该应用的输入为带有井名，日期，日产量或月产量的数据，注入井数据为可选输入;
            如果未能自动识别读取文件中的输入数据，需要用户自行设置需读取数据的表头;
            读取成功后，用户可自行选择预测函数的滞后阶数，拟合原理详见用户手册;
            用户也可通过滞后阶数分析选项选取最优滞后阶数;
            在拟合完成后可进行脉冲响应分析，该选项可分析注入井对采出井的影响程度;
            '''
            tkinter.messagebox.showinfo('帮助', temp_info)

        menu_bar = tkinter.Menu(self.root)
        self.root.config(menu=menu_bar)
        help_menu = tkinter.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="关于", command=get_help)
        menu_bar.add_cascade(label="帮助", menu=help_menu)

        self.root.mainloop()

    def read_csv_P(self, ):
        '''
        读取采出井报告数据,若其中包含年月,井号,日采油量等表头则设为默认表头
        :return:
        '''
        from util import _read
        self.Pdata = _read()
        col = list(self.Pdata.columns)
        self.P_cl_1['values'] = col
        self.P_cl_2['values'] = col
        self.P_cl_3['values'] = col
        try:
            self.P_cl_1.current(col.index('年月'))
            self.P_cl_2.current(col.index('井号'))
            self.P_cl_3.current(col.index('日产油量'))
        except:
            self.P_cl_1.current(0)
            self.P_cl_2.current(0)
            self.P_cl_3.current(0)
        self.validate_command['state'] = tkinter.NORMAL
        self.lag_command['state'] = tkinter.NORMAL
        self.predict_command['state'] = tkinter.NORMAL

    def read_csv_I(self, ):
        '''
        读取注入井报告数据,若其中包含年月,井号,日注入量等表头则设为默认表头
        :return: 原始注入井报告数据
        '''
        from util import _read
        self.Idata = _read()
        col = list(self.Idata.columns)
        self.I_cl_1['values'] = col
        self.I_cl_2['values'] = col
        self.I_cl_3['values'] = col
        try:
            self.I_cl_1.current(col.index('年月'))
            self.I_cl_2.current(col.index('井号'))
            self.I_cl_3.current(col.index('日注入量'))
        except:
            self.I_cl_1.current(0)
            self.I_cl_2.current(0)
            self.I_cl_3.current(0)

    def data_prepro(self):
        '''
        该方法将油田的日报/月报数据转换为日期/井号格式的表格数据
        :return: 日期/井号格式的表格数据
        '''
        m_assert_error([self.Pdata is not None, self.P_cl_1.get(), self.P_cl_2.get(), self.P_cl_3.get()],
                       ['未读取采出井数据', '未选取采出井年月表头', '未选取采出井井号表头', '未选取采出井流量表头'])
        np.random.seed(1)
        Ptimeseries, Pdata = table2timeseries(self.Pdata, self.P_cl_1.get(), self.P_cl_2.get(), self.P_cl_3.get(),
                                              'prod')
        if self.v2.get() == 1:
            m_assert_error([self.Idata is not None, self.I_cl_1.get(), self.I_cl_2.get(), self.I_cl_3.get()],
                           ['未读取注入井数据', '未选取注入井年月表头', '未选取注入井井号表头', '未选取注入井流量表头'])
            Itimeseries, Idata = table2timeseries(self.Idata, self.I_cl_1.get(), self.I_cl_2.get(), self.I_cl_3.get(),
                                                  'inj')
            I_mintime = min(Idata[self.I_cl_1.get()])
            P_mintime = min(Pdata[self.P_cl_1.get()])
            if I_mintime > P_mintime:
                skip_i = list(Ptimeseries.index).index(I_mintime)
                Ptimeseries = Ptimeseries[skip_i:]
            else:
                skip_i = list(Itimeseries.index).index(P_mintime)
                Itimeseries = Itimeseries[skip_i:]
            self.total = pd.concat((Itimeseries, Ptimeseries), axis=1)
            self.icol_num = np.array([list(self.total.columns).index(c) for c in self.total.columns if 'inj' in c])
        else:
            self.total = Ptimeseries

        self.total.index = pd.to_datetime(self.total.index.astype(str), format='%Y%m')
        self.total_mean = self.total.rolling(window=4).mean()[4:]
        self.pcol_num = np.array([list(self.total.columns).index(c) for c in self.total.columns if 'prod' in c])

        #self.skip_i = int(self.year_cl.get())
        #m_assert_error(self.skip_i < len(self.total) - self.total.shape[1], '跳过步数必须小于时间步数')

        self.maxlags = int(self.Lag_cl.get())

    def prepare_Y(self, skip_i, part_i):
        '''
        该方法为预处理步骤，包括log1p化及平滑化
        :param skip_i: 跳过的时间步长数
        :param part_i: 若是验证模型，则该值默认为n-10,预测则为n
        :return:
        '''
        Y_endog = self.total_mean[skip_i:part_i].iloc[:, self.pcol_num]
        Y_endog = pd.DataFrame(np.log1p(Y_endog), columns=Y_endog.columns, index=Y_endog.index)
        Y_endog += np.random.rand(*Y_endog.shape) / 1e10
        Y_future = self.total_mean[part_i:].iloc[:, self.pcol_num]
        if self.v2.get():
            Y_exog = np.log1p(self.total_mean[skip_i:part_i].iloc[:, self.icol_num])
            Y_exog += np.random.rand(*Y_exog.shape) / 1e10
            Y_exog_future = np.log1p(self.total_mean[part_i:].iloc[:, self.icol_num])
            Y_exog_future += np.random.rand(*Y_exog_future.shape) / 1e10
        else:
            Y_exog = None
            Y_exog_future = None
        return Y_endog, Y_future, Y_exog, Y_exog_future

    def maxlag_selection(self):
        '''
        最大滞后阶数选择
        :return: 不同准则下最优滞后阶数
        '''
        self.data_prepro()
        Y, Y_future, Y_exog, Y_exog_future = self.prepare_Y(0, -10)

        model = VAR(endog=Y, exog=Y_exog)
        ics = defaultdict(list)
        for p in range(self.maxlags + 1):
            result = model._estimate_var(p, offset=self.maxlags - p)
            for k, v in iteritems(result.info_criteria):
                ics[k].append(v)
        selected_orders = dict((k, np.array(v).argmin())
                               for k, v in iteritems(ics))
        t_str = get_ic_table(ics, selected_orders)
        t_str += '\n%s' % str(selected_orders)
        self.model = model
        self.InfoText.delete(0.0, tkinter.END)
        self.InfoText.insert(0.0, t_str)

    def fit_and_validate(self, ):
        '''
        拟合并验证
        :return:
        '''
        from m_VAR import m_VAR_model
        from util import re_log1p
        self.data_prepro()
        l2=int(self.re_cl.get())
        steps_valid = 10
        Y_valid, Y_future, Y_exog, Y_exog_future = self.prepare_Y(0, -steps_valid)
        model = m_VAR_model(Y_valid, exog=Y_exog, maxlags=self.maxlags, l2=l2)
        model.fit()
        self.model = model
        y_pred=predict_fix(model.forecast(Y_valid, steps_valid, exog_future=Y_exog_future, ))

        score_mae, score_mr=get_score(y_pred=y_pred, y_true=Y_future)
        if self.v2.get():
            self.verbose_list.insert(0, '滞后阶数:%d, 正则化系数:%d, 考虑注入井, mae:%.6f, mer:%.6f' % (
            int(self.Lag_cl.get()), l2, score_mae, score_mr))
        else:
            self.verbose_list.insert(0, '滞后阶数:%d, 正则化系数:%d, mae:%.6f, mer:%.6f' % (
            int(self.Lag_cl.get()), l2, score_mae, score_mr))
        if self.v3.get():
            point_forecast, forc_lower, forc_upper = model.forecast_interval(np.array(Y_valid), steps_valid,
                                                                             exog_future=Y_exog_future, )
            forc_lower = re_log1p(forc_lower)
            forc_upper = re_log1p(forc_upper)
            self.validate_window = m_forecast_plot_window(Y_valid, Y_future, y_pred, fore_cov=(forc_lower, forc_upper))
        else:
            self.validate_window = m_forecast_plot_window(Y_valid, Y_future, y_pred, )

    def fit(self, ):
        '''
        拟合并预测
        :return:
        '''
        from m_VAR import m_VAR_model
        from util import re_log1p
        self.data_prepro()
        l2 = int(self.re_cl.get())
        steps_valid = len(self.total_mean)
        Y_endog, _, Y_exog, _ = self.prepare_Y(0, steps_valid)
        model = m_VAR_model(Y_endog, exog=Y_exog, maxlags=self.maxlags, l2=l2)
        model.fit()
        self.model = model
        if self.v2.get():
            self.verbose_list.insert(0, '滞后阶数:%d, 正则化系数:%d, 考虑注入井, 拟合完成' % (
            int(self.Lag_cl.get()),l2))
        else:
            self.verbose_list.insert(0, '滞后阶数:%d, 正则化系数:%d, 拟合完成' % (
            int(self.Lag_cl.get()), l2))
    def predict(self):
        pass
        '''
        
        :return: 
        '''

    def m_irf(self):
        '''
        拟合完成后，进行脉冲响应分析可对注水井进行评价
        :return: 脉冲响应分析窗口
        '''
        m_assert_error([self.model is not None, self.icol_num is not None],
                       ['需要才可进行脉冲响应分析', '脉冲响应分析需要输入注入井数据'])
        self.irf_window = m_irf_window(self.model, self.total_mean, self.pcol_num, self.icol_num)


if __name__ == "__main__":
    win = window()
