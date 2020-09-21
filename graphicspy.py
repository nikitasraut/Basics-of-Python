import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

iris=pd.read_csv("G:/Statistics (Python)/Datasets/iris.csv")

########################### Using pandas #############################
iris['Sepal.Length'].plot(kind='hist')
plt.show()

iris['Sepal.Length'].plot(kind='hist',color="purple",bins=20)
plt.show()

iris['Sepal.Length'].plot(kind='hist',color="purple",
                          alpha=0.2, bins=20)
plt.show()

iris['Sepal.Length'].plot(kind='hist',color="purple",
    alpha=0.4,bins=20)
plt.xlabel('Sepal Length')
plt.title('Histogram')
plt.show()

Orders = pd.read_csv("G:/Statistics (Python)/Datasets/Orders.csv")
cts = Orders['Payment Terms'].value_counts()
cts.plot(kind='bar')
plt.show()

cts = Orders['Payment Terms'].value_counts()
cts.plot(kind='bar', color="violet")
plt.show()

cts.plot(kind='pie')
plt.show() 
 
iris['Sepal.Length'].plot(kind='kde')
plt.show()

iris.plot(kind='scatter', x='Sepal.Length', 
          y='Sepal.Width')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Scatter Plot")
plt.show()

iris['Sepal.Length'].plot(kind='box')
plt.show()

iris.plot(kind='box')
plt.show()

############################# Using matplotlib ######################

x = [1990,1995,2000,2005,2010,2015]
y = [800,345,125,423,560,782]

# How not to plot
plt.bar(x, y, align='center', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Sales ('000)")
plt.title('Sales Comparison')
plt.show()

# How to plot
x = [str(v) for v in x]
plt.bar(x, y, align='center', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Sales ('000)")
plt.title('Sales Comparison')
plt.show()

# How to plot
y_pos = np.arange(len(x))
plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, x)
plt.xlabel("Year")
plt.ylabel("Sales ('000)")
plt.title('Sales Comparison')
plt.show()

x = [1990,1995,2000,2005,2010,2015]
y = [800,345,125,423,560,782]
plt.plot(x,y)
plt.xlabel("Year")
plt.ylabel("Sales ('000)")
plt.title('Sales Analysis')
plt.show()

x = [1990,1995,2000,2005,2010,2015]
y = [800,345,125,423,560,782]
plt.scatter(x,y)
plt.show()

### Histogram
plt.hist(iris['Sepal.Length'] , bins=10)
plt.xlabel("Sepal Length")
plt.title("Sepal Length of Iris")
plt.show()

plt.hist(iris['Sepal.Length'] , bins=10, alpha=0.3)
plt.xlabel("Sepal Length")
plt.title("Sepal Length of Iris")
plt.show()

plt.hist(iris['Sepal.Length'],color='pink')
plt.title("Sepal Length of Iris")
plt.show()

plt.hist(iris['Sepal.Length'],color='purple',
         histtype='step')
plt.title("Sepal Length of Iris")
plt.show()


cts = Orders['Payment Terms'].value_counts()
plt.bar(cts.index,cts)
plt.title("Modes of Payment")
plt.show()

#OR

cts = Orders['Payment Terms'].value_counts()
plt.bar(['Online','Cheque','Cash'],cts)
plt.title("Modes of Payment")
plt.show()


#OR

cts = Orders['Payment Terms'].value_counts()
ind = np.arange(3)
plt.bar(ind,cts)
plt.xticks(ind,('Online','Cheque','Cash'))
plt.title("Modes of Payment")
plt.show()

plt.boxplot(iris['Sepal.Length'])
plt.show()

plt.scatter(iris['Sepal.Length'], 
            iris['Sepal.Width'])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title('Scatter Plot')
plt.show()

## Customizing the axes
plt.scatter(iris['Sepal.Length'], iris['Sepal.Width'])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title('Scatter Plot')
plt.axis((4,9,1,5))
plt.show()

###################################################

plt.subplot(1,2,1)
plt.hist(iris['Sepal.Length'],color='red')
plt.title("Sepal Length of Iris")

plt.subplot(1,2,2)
plt.hist(iris['Sepal.Width'])
plt.title("Sepal Width of Iris")

#plt.tight_layout()
plt.show()

############################### Seaborn ###############################

import seaborn as sns
sns.countplot('Payment Terms',data=Orders)
plt.show()

g = sns.catplot('Payment Terms',data=Orders, kind="count")
plt.show()

sns.boxplot(y='Sepal.Length', data=iris)
plt.show()

sns.boxplot(x='Species', y='Sepal.Length', data=iris)
plt.show()

sns.distplot(iris['Sepal.Length'])
plt.show()

sns.regplot(x='Sepal.Length', y='Sepal.Width', data=iris, 
            fit_reg=False)
plt.show()

#sns.set(color_codes=True)
sns.lmplot(x='Sepal.Length', y='Sepal.Width', data=iris,
            fit_reg=False)
plt.show()

sns.lmplot(x='Sepal.Length', y='Sepal.Width', data=iris, hue='Species',
            fit_reg=False)
plt.show()

sns.regplot(x='Sepal.Length', y='Sepal.Width', data=iris)
plt.show()

# Facets

g = sns.FacetGrid(iris, col="Species")
g = g.map(plt.hist, "Sepal.Length")
plt.show()

g = sns.FacetGrid(iris, row="Species")
g = g.map(plt.hist, "Sepal.Length")
plt.show()

g = sns.FacetGrid(iris, col="Species")
g = g.map(plt.scatter, "Sepal.Length","Sepal.Width")
plt.show()

# JointPlot

sns.jointplot(x='Sepal.Length', y='Sepal.Width', data=iris)
plt.show()

sns.jointplot(x='Sepal.Length', y='Sepal.Width', data=iris,kind="reg")
plt.show()

# Pairs Plot / Matrix Plot

sns.pairplot(iris)
plt.show()


sns.pairplot(iris, hue="Species")
plt.show()

# Heatmap
sns.heatmap( iris.corr(),annot=True)
plt.show()

sns.heatmap( iris.corr(),cmap="YlGnBu")
plt.show()


