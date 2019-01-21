import pandas as pd
def table2timeseries(data,cl_1,cl_2,cl_3,fix):
    data = data.sort_values(by=cl_1)
    data.loc[data[cl_3] < 0, cl_3] = 0
    timeseries = data.pivot_table(columns=cl_2, index=cl_1,
                                    values=cl_3, fill_value=0)
    timeseries.columns = ['%s_%s' % (c,fix) for c in timeseries.columns]
    return timeseries,data