# Pandas is an extremely convenient and intuitive Python library for handling data. 
#The most common objects used in Pandas are dataframes: data tables with header and index.

import pandas as pd
import numpy as np

dates = pd.date_range('20190101', periods = 12)
df = pd.DataFrame(np.random.randn(12, 3), index = dates, columns = list('abc'))

df

df2 = pd.DataFrame({ 'A' : 1.,
 'B' : pd.Timestamp('20130102'),
 'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
 'D' : np.array([3] * 4,dtype='int32'),
 'E' : pd.Categorical(["test","train","test","train"]),
 'F' : 'foo' })
 
df2
