# Ex02-Outlier
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

(i) Using IQR detect weight outliers and print them

(ii) Using IQR, detect height outliers and print them

# Aim:
TO detect and remove the outliers in the given data set and save the final data.

# EXPLANATION
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

## ALGORITHM
STEP 1 Read the given Data

STEP 2 Get the information about the data

STEP 3 Detect the Outliers using IQR method and Z score

STEP 4 Remove the outliers

# CODE 
### bhp.csv
```
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("bhp.csv")
q1 = df['price_per_sqft'].quantile(0.25)
q2 = df['price_per_sqft'].quantile(0.5)
q3 = df['price_per_sqft'].quantile(0.75)
iqr = q3-q1
iqr
low = q1-1.5*iqr
low
high = q3+1.5*iqr
high
df = df[((df['price_per_sqft']>=low) & (df['price_per_sqft']<=high))]
df
z = np.abs(stats.zscore(df['price_per_sqft']))
z
df1 = df[z<3]
df1
```
### height_weight.CSV
```
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("height_weight.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
low = q1 - 1.5*iqr
low
high = q3 + 1.5*iqr
high
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
z = np.abs(stats.zscore(df['height']))
z
df1 = df[z<3]
df1
```
# OUTPUT
### bhp.csv
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/2bb4bf97-b308-435c-917a-502bd1ed4f88)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/e356ac1e-be13-4309-98a4-4acc0e040e12)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/41045de4-50c6-4438-84d9-d60cfde78716)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/7368fa6e-8d91-470d-84a2-6a452caee25d)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/1205400e-72eb-4534-8400-fd24928d6b08)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/c3fbf939-9606-4560-9ff3-b2c5b9441a0f)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/70a38680-c475-4eea-b086-be320bc6dc36)
### height_weight
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/1b5054b5-66ec-451b-b44e-2e6ad997a59e)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/8a07a5d1-f27c-4cbb-b647-39e5b44cf9b7)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/5ba40b64-0a20-4d11-88bf-15591ea80026)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/89072b3b-da25-4614-b429-b93c664c4ffb)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/e2db5335-8652-4a31-9cef-b58b96632889)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/2f95f644-4442-4664-ad5d-5942c3e5dc74)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/86bd7416-d02a-45e7-9128-6da5b9413c1c)
![image](https://github.com/gowrisankarponnusamy/ODD2023---Datascience---Ex-02/assets/119393123/8221b6f6-b55e-4226-8f6f-96bb6409a033)

# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.
