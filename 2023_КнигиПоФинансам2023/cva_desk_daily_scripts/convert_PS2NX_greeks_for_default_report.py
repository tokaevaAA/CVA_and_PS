import numpy as np
import datetime as dt
import sys
import pandas as pd
pd.options.mode.chained_assignment = None # default='warn'

import ps, ps_utils
ps_utils.switch_to_dev()

def get_report(dfps):
    
    dfps_bycpty = dfps[dfps['ContractID']=='Total']
    
    def getdf(mytype):
        
        dfps_mytype = dfps_bycpty[['Metric', 'Cpty', mytype]]
        
        otv = None
        
        for el in dfps_mytype['Metric'].unique():       
            
            d = dfps_mytype[dfps_mytype['Metric'] == el]
            d['Metric'] = [mytype for i in range(len(d))]
            d.rename(columns = {d.columns[2]:el}, inplace=True)
            
            if otv is None:
                otv = d
            else:
                otv = otv.merge(d, on=['Metric','Cpty'], how='outer')

        return otv

    df_pv = getdf('ProductValue')
    df_cva = getdf('CVA')
    df_dva = getdf('DVA')
    df_bcva = getdf('BCVA')
    
    otv = pd.concat([df_cva, df_dva, df_bcva], axis=0)

    return otv

def get_tenor(date_a, date_b):
    
    a = (date_b - date_a).days
    a_w = a / 7
    a_m = a / 30
    a_y = a / 365
    if a < 7:
        return "{}D".format(round(a))
    elif a >= 7 and a < 28:        
        return "{}W".format(round(a_w))
    elif a >= 28 and a_m < 12:
        return "{}M".format(round(a_m))        
    elif (a_m >= 12) and a_y < 1.08:
        return "{}Y".format(round(a_y))
    elif a_y >= 1.08 and a_y < 1.8:
        return "{}M".format(round(a_m))
    elif a_y > 1.8:
        return "{}Y".format(round(a_y))


fname = sys.argv[1]

run_id = fname.split(".")[0].split("_")[-1]

run = ps.get(run_id)
run_sample = run['Result'][0]
run_sample['Request']['Model']['MarketDataSet']['AsOfDate']
run_dt = run_sample['Request']['Model']['MarketDataSet']['AsOfDate'].date()
run_dt_str = run_sample['Request']['Model']['MarketDataSet']['AsOfDate'].date().strftime("%Y-%m-%d")

out_fname = "Greeks_" + run_dt_str + "_" + run_id + ".xlsx"

print("Reading data from %s" % fname)

metrics = pd.read_excel(fname)
print("Len=",len(metrics))

BCVA = metrics.loc[(metrics['Metric'] == 'Base') & (metrics['ContractID'] == 'Total'), 'BCVA'].sum()
print("Total BCVA is %0.2f" % BCVA)

print("Converting Greeks to required format")
report_out = get_report(metrics)

metric_names = report_out.columns
new_columns = []

for i in range(len(metric_names)):
    metric_name = metric_names[i]
    if 'CS01_buckets' in metric_name:        
        a = metric_name.split(".")
        tenor_date = dt.datetime.strptime(a[-1], '%Y-%m-%d').date()
        a[-1] = get_tenor(run_dt, tenor_date)
        metric_name = ".".join(a)
    elif 'DV01_buckets' in metric_name:
        a = metric_name.split(".")
        tenor_date = dt.datetime.strptime(a[-1], '%Y-%m-%d').date()
        a[-1] = get_tenor(run_dt, tenor_date)
        metric_name = ".".join(a)       
    new_columns.append(metric_name)

report_out.set_axis(new_columns, axis='columns', inplace=True)

print("Writing Greeks in converted format to %s" % out_fname)
report_out.to_excel(out_fname, index=False)
