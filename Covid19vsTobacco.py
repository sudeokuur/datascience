# This source code includes a Covid-19 dataset which I analyzed, visualized.and
# I found a ratio which summarise the relationship between Covid-19 and the usage of tobacco.
# Dataset link -> https://www.kaggle.com/datasets/meirnizri/covid19-dataset 

#importing necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#importing the dataaset
df = pd.read_csv('https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data.csv')
#general look to the dataframe
df.info

#to see the first 5 rows of our dataframe
df.head()

#because I want to see all my data clearly, I changed the numeric data about the date of time as died/not died.
transformedData = df
transformedData.loc[transformedData['DATE_DIED'] =='9999-99-99', 'DIED'] = "NOT DIED"
transformedData.loc[transformedData['DATE_DIED'] !='9999-99-99', 'DIED'] = "DIED"

#to see if my operation done successfully
transformedData[['DATE_DIED', 'DIED']].head(10)

#I made a histogram to see the tobacco & &dead ratio clearly
transformedData = transformedData.loc[(transformedData.TOBACCO != 98)]
d = sns.histplot(data=transformedData, x='TOBACCO', hue='DIED')
plt.show(d)


num_tobacco = transformedData['TOBACCO'].loc[(transformedData.TOBACCO == 1)].count()
num_tobacco_died = transformedData['TOBACCO'].loc[(transformedData.TOBACCO == 1) &(transformedData.DIED == 'DIED')].count()

percTobDeath = (num_tobacco_died/num_tobacco)*100
print(percTobDeath)

# final ratio 
print("Percentage of tobacco users dead: ", round(percTobDeath), "%")

