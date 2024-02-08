# Using K-Means Clustering for Customer Segmentation

## About

A marketing department are looking to refine their advertising campaigns. In order to do this they would like to segment their existing consumer base. The department could manually construct customer segments but would like a reproducible method for future campaigns.

In this project we will create an unsupervised machine learning algorithm in Python to segment customers. Creating a K-Means Clustering algorithm to group customers by commonalities and provide the marketing department with insights into the different types of customers they have.

Skills Showcased:

-   Unsupervised Machine Learning - K-Means Clustering
-   Data Cleaning
-   Feature Engineering
-   Data Analysis

[View more projects like this!](https://cian-murray-doyle.github.io/)

## Libraries Overview

The following Python libraries will be used for this project.

``` python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

If these libraries are not installed, they can be installed with the following code.

``` python
# Install NumPy
!pip install numpy

# Install Pandas
!pip install pandas

# Install Seaborn
!pip install seaborn

# Install scikit-learn (includes KMeans)
!pip install scikit-learn

# Install Matplotlib
!pip install matplotlib
```

## Preparing the Data

Before any algorithms are applied, the dataset needs to be cleaned.

![Alt Text](images/na_vals_table.png)

### Missing Values

Some examples of dealing with missing values are; removing them completely, replacing them with the mean/median or inferring the missing values. Due to the k-means clustering's sensitivity to missing values, our example will opt to remove all missing values from the dataset.

### Outliers

To detect outliers we will use `online_sales.describe()`. In our example, we can see there are negative values for “Quantity” and values of 0 for “UnitPrice”, as we know these are not possible values and therefore need to be removed.

Currently with our data we do not have enough context to begin removing outliers using IQR or Z-Scores, we will remove these later on in the process.

### Feature Engineering

In order to keep refining our data for the model we need to gain new insights from out data. This will be done by creating new features and using and converting the data to an RFM table.

### Irrelevant Features

``` python
online_sales["TotalCost"] = online_sales["Quantity"]\*online_sales["UnitPrice"]

correlation_matrix = online_sales[["Quantity","UnitPrice","TotalCost"]].corr() sns.heatmap(correlation_matrix, vmin=0.0, vmax=1.0, annot=True) plt.title("Correlation Matrix",fontweight="bold") plt.show()
```

![](images/corr_matrix.png)

The heatmap shows us which features have a high correlation with one another, in our case we can use `"TotalCost"` rather than `"Quantity"` and `"Price"` when making our table/model.

### Formatting
