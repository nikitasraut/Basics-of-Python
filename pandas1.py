demog = { 'India': { 'area':3287263, 'population':1339180127 },
           'China': { 'area':9596961, 'population':1409517397 },
           'US': { 'area':9833520, 'population':324459463 },
           'Indonesia': { 'area':1904569, 'population':263991379 } }

import pandas as pd

topPop = pd.DataFrame(demog)

iris = pd.read_csv(r"C:\Users\admin\Desktop\PYTHON\Datasets\iris.csv")

iris["Sepal.Width"]

topPop["India"]
topPop[["India"]]

type(topPop["India"])
type(topPop[["India"]])
iris.columns

topPop[["India","China"]]

iris[2:4]

iris[:3]

## Using loc
topPop.loc["area"]
topPop.loc[["area"]]
type(topPop.loc["area"])
type(topPop.loc[["area"]])

topPop.loc[["area","population"]]

topPop.loc[["area","population"],["India","US"]]
topPop.loc[["area"],["India","US"]]
topPop.loc[:,["India","US"]]

topPop.iloc[[1],[1,3]]
topPop.iloc[:,[1,3]]

iris[iris["Sepal.Width"] > 3.9]

iris[(iris['Sepal.Length']>5) & (iris['Petal.Width']>2)]

## Doesn't work in Python
iris["Sepal.Width"] > 3.5 & iris["Sepal.Length"] > 5.2
    
## Works
import numpy as np
np.logical_not(iris["Species"]=="setosa")
np.logical_and(iris["Sepal.Width"] > 3.5 , iris["Sepal.Length"] > 5.2)
np.logical_or(iris["Sepal.Width"] > 3.5 , iris["Sepal.Length"] > 5.2)

iris[np.logical_and(iris["Sepal.Width"] > 3.5 , iris["Sepal.Length"] > 5.2)]

# This also works
iris.loc[(iris['Sepal.Length']>5) & (iris['Petal.Width']>1)]

# Listing column names
iris.columns

# Renaming the columns
iris = iris.rename(columns={'Sepal.Length': 'Sepal Length', 
                     'Sepal.Width': 'Sepal Width',
                     'Petal.Length': 'Petal Length',
                     'Petal.Width':'Petal Width'})

# OR

iris.rename(columns={'Sepal.Length': 'Sepal Length', 
                     'Sepal.Width': 'Sepal Width',
                     'Petal.Length': 'Petal Length',
                     'Petal.Width':'Petal Width'},inplace=True)

print(iris.info())

iris['Sepal Width'] = iris['Sepal Width'].astype(str)

# Handling Categories
iris['Species'] = iris['Species'].astype('category')
iris['Species'].cat.categories
iris['Species'] = iris.Species.cat.reorder_categories(['virginica', 'setosa', 'versicolor' ], ordered=True)
iris['Species'].cat.categories

# Handling Dates
ords = pd.read_csv(r"C:\Users\admin\Desktop\PYTHON\Datasets\Orders.csv")
ords.dtypes
ords['Order Date'] = pd.to_datetime(ords['Order Date'],format="%d-%b-%y")
ords.dtypes
 
#OR

ords = pd.read_csv(r"C:\Users\admin\Desktop\PYTHON\Datasets\Orders.csv",
                   parse_dates=['Order Date'])
ords.dtypes

ords['year'] = ords['Order Date'].dt.year
ords['month'] = ords['Order Date'].dt.month
ords['day'] = ords['Order Date'].dt.day

###Decision Control
x = 35
if x < 30 :
    print("Less")
elif x < 40 :
    print("Medium")
else :
    print("High")
    
x = 20
if x < 30 :
    print("Less")
    y = x+56
elif x < 40 :
    print("Medium")
    y = x+96
else :
    print("High")
    y = x-90

###Loop Control

f = 97
while f < 100:
    print("Increasing..")
    f = f + 1
    print(f)
print("Loop Over")

fam = [12.3,34.5,23.5,20.4]
for i in fam:
    print(i*i)
    
for index, height in enumerate(fam) :
    print("index " + str(index+1) + ": " + str(height))

house = [["Hallway", 10.25], 
         ["Kitchen", 19.0], 
         ["Living Room", 20.0], 
         ["Bedroom", 9.75], 
         ["Bathroom", 9.55]]
         
# Build a for loop from scratch
for x in house:
    print("the " + str(x[0]) + " is " + str(x[1]) + " sqm")
    
for ch in "statistics":
    print(ch.capitalize())

cap = ""   
for ch in "statistics":
    cap = cap + ch.capitalize()
    
demog = { 'India': { 'area':3287263, 'population':1339180127 },
           'China': { 'area':9596961, 'population':1409517397 },
           'US': { 'area':9833520, 'population':324459463 },
           'Indonesia': { 'area':1904569, 'population':263991379 } }

for key, value in demog.items() :
    print(key + " has " + str(value))
    
# Also
for dd, ff in demog.items() :
    print(dd + " has ", end="")
    d,f = ff.items()
    print(d[0], d[1], end = " and ")
    print(f[0], f[1])
    
# =============================================================================
# print(topPop)
# 
# for i in topPop:
#     print(i)
# =============================================================================
    

telecom = pd.read_csv(r"C:\Users\admin\Desktop\PYTHON\SP Machine Learning\Cases\Telecom\Telecom.csv")
telecom.head()    

pd.crosstab(index=telecom['Response'],columns="count")

pd.crosstab(index=telecom['Response'],columns=telecom["Gender"])
