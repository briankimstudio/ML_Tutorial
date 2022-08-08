#
# Visualization
#

import seaborn as sns
import matplotlib.pyplot as plt

my_dataset = sns.load_dataset('iris')

print(f'\nData structure\n')
print(my_dataset.info())

print(f'\nData preview\n')
print(my_dataset.head())

print(f'\nCorelation\n')
print(my_dataset.corr())

#
# Pair plot
#
sns.pairplot(my_dataset)
plt.show()

#
# Pair plot by class
#
sns.pairplot(my_dataset, hue='species')
plt.show()

#
# Scatter plot for two features
#
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=my_dataset)
plt.show()

#
# Line plot
#
sns.lineplot(x='petal_length', y='petal_width', data=my_dataset)
plt.title('Line plot')
plt.show()

sns.boxplot(x='species', y='petal_width', data=my_dataset)
plt.title('Box plot')
plt.show()

sns.violinplot(x='species', y='petal_width', data=my_dataset)
plt.title('Violin plot')
plt.show()

sns.stripplot(x='species', y='petal_width', data=my_dataset)
plt.title('Violin plot')
plt.show()

sns.swarmplot(x='species', y='petal_width', data=my_dataset)
plt.title('Violin plot')
plt.show()

#
# Bar plot
#
sns.barplot(x='species', y='petal_width', data=my_dataset)
plt.title('Bar plot')
plt.show()

sns.countplot(x='species', data=my_dataset)
plt.title('Count plot')
plt.show()


#
# Distribution
#
sns.distplot(my_dataset['sepal_length'])
plt.show()

sns.jointplot(x='species', y='sepal_width', data=my_dataset)
plt.show()

#
# Regression plot
#
my_dataset = sns.load_dataset('tips')
sns.lmplot(x='total_bill', y='tip', data=my_dataset)
plt.show()

#
# Heatmap
#
sns.heatmap(my_dataset.corr())
plt.show()