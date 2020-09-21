import pandas as pd
import numpy as np

job = pd.read_csv("G:/Statistics (Python)/Datasets/JobSalary.csv")

# Detecting Null columns
null_columns=job.columns[job.isnull().any()]
null_columns

# Number of missings in each column
job[null_columns].isnull().sum()

# OR
np.sum(pd.isnull(job))

# OR
job.isnull().sum()

# Dropping the specific rows
job.drop([2,3,6],axis=0)

# Dropping the specific columns
job.drop(['Computer'],axis=1)

mu_comp = job['Computer'].mean()
job['a_comp'] = job['Computer'].fillna(mu_comp)
job

job['a_comp_ff'] = job['Computer'].ffill()
job

job['a_comp_ff'] = job['Computer'].bfill()
job

boston = pd.read_csv("G:/Statistics (Python)/Datasets/Boston.csv")

# Column-wise mean
boston.apply(np.mean, axis=0)

# Row-wise mean
boston.apply(np.mean, axis=1)

quality = pd.read_csv("G:/Statistics (Python)/Datasets/quality.csv")
qual_melt = pd.melt(quality, id_vars='Sno')

qual_pivot = pd.pivot_table(qual_melt, index='Sno',
                            columns='variable',values='value')

grp_by_var = qual_melt.groupby('variable')
grp_by_var['value'].mean()

#OR

qual_melt.groupby('variable')['value'].mean()

qual_melt.groupby('variable')['value'].std()

telecom = pd.read_csv("G:/Statistics (Python)/Cases/Telecom/Telecom.csv")

telecom['Response']
telecom['Response'].value_counts()

pd.crosstab(index=telecom['Response'],columns='Count')

pd.crosstab(index=telecom['Response'],
            columns=telecom['Gender'])

### Contingency Chi-square
from scipy import stats
stats.chi2_contingency(pd.crosstab(index=telecom['Response'],
            columns=telecom['Gender']))


pd.crosstab(index=telecom['Response'],columns=telecom['Gender'],
            margins=True)

### Proportions
pd.crosstab(index=telecom['Response'],columns=telecom['Gender'],
            margins=True,normalize='all')

### Row Proportions
pd.crosstab(index=telecom['Response'],columns=telecom['Gender'],
            margins=True,normalize='index')

### Column Proportions
pd.crosstab(index=telecom['Response'],columns=telecom['Gender'],
            margins=True,normalize='columns')


diamonds = pd.read_csv("G:/Statistics (Python)/Datasets/diamonds.csv")

pd.crosstab(index=diamonds['cut'],columns='count')
pd.crosstab(index=diamonds['color'],columns='count')

pd.crosstab(index=diamonds['cut'],columns=diamonds['color'])
pd.crosstab(index=diamonds['cut'],columns=diamonds['color'],margins=True)

###############################################################################
##### Bucketizing the data######
cct = pd.cut(boston["medv"], 3)
type(cct)
np.unique(cct,return_counts=True)
#OR
cct = pd.Series(cct)
cct.value_counts()

bins = pd.IntervalIndex.from_tuples([(4.9, 13), (13, 30), (30, 51)])
cct = pd.cut(boston["medv"], bins)
cct.value_counts()
