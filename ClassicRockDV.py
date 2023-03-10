# dataset link -> https://www.kaggle.com/datasets/juliotorniero/classic-rock-top-500-songs
# I visualized & analyzed this dataset in a general look.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
#importing the dataaset
df = pd.read_csv(' https://www.kaggle.com/datasets/juliotorniero/classic-rock-top-500-songs')
df.head()


df.shape
df.dtypes


df.describe()
df["Artist"].unique()


#visualization via top 10 artist
figure,axis=plt.subplots(nrows=2,ncols=1,figsize=(10,20))
d = df.groupby("Artist")
counts=df.Artist.value_counts()[:10]
avg_year=d.Year.mean().sort_values()[:10]


ax0=sns.barplot(y=counts.index, x=counts.values,orient="horizontal",ax=axis[0])
ax1=sns.barplot(y=avg_year.index, x=avg_year.values,orient="horizontal",ax=axis[1])

axis[0].set(xlabel='count',title='10 Artist with maximum songs')
ax0.bar_label(ax0.containers[0])

axis[1].set(xlabel='average year',title='10 Average Year')
ax1.bar_label(ax1.containers[0])


plt.ylabel('Artist')
sns.despine()
plt.show()

#heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)


#histplot
sns.histplot(df['Year'])

#scatter plot
plt.scatter(data=df,x="Year",y="2022")




