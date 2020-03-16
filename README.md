
# Confidence Intervals with T Distribution - Lab

## Introduction

In the previous lab, we saw that if we have the standard deviation for the population, we can use use $z$-score to calculate our confidence interval using the mean of sample means. 

If, on the other hand, the standard deviation of the population is not known (which is usually the case), you have to use the standard deviation of your sample as a stand-in when creating confidence intervals. Since the sample standard deviation is often different than that of the population, further potential errors are introduced to our confidence intervals. To account for this error, we use what's known as a t-critical value instead of the $z$-critical value.

The t-critical value is drawn from what's known as a t-distribution.
> A t-distribution  closely resembles the normal distribution but gets wider and wider as the sample size falls.

<img src="images/new_t-distr-img.png" width="500">

The t-distribution is available in `scipy.stats` with the nickname "t" so we can get t-critical values with `stats.t.ppf()`.

## Objectives
You will be able to:

* Calculate confidence intervals
* Interpret confidence intervals in relation to true population parameters

## Let's get started!


```python
# Import the necessary libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import math
```

Let's investigate point estimates by generating a population of random age data collected at two different locations and then drawing a sample from it to estimate the mean:


```python
np.random.seed(20)
population_ages1 = np.random.normal(20, 4, 10000) 
population_ages2 = np.random.normal(22, 3, 10000) 
population_ages = np.concatenate((population_ages1, population_ages2))

pop_ages = pd.DataFrame(population_ages)
pop_ages.hist(bins=100,range=(5,33),figsize=(9,9))
plt.title('Histogram of the Population Ages')

pop_ages.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21.008578</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.671277</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.784588</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.662256</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.163276</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>23.489438</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36.140116</td>
    </tr>
  </tbody>
</table>
</div>




![png](index_files/index_6_1.png)


Let's take a new, smaller sample (of size smaller than 30) and calculate how much the sample mean differs from the population mean.


```python
np.random.seed(23)

sample_size = 25
sample = pop_ages.sample(n=25) # Take a random sample of size 25 from above population
sample_mean = sample.mean()  # Calculate sample mean 

# Print sample mean and difference of sample and population mean 

# Sample Mean: 19.870788629471857
# Mean Difference: 1.1377888781920937
sample_mean[0], pop_ages.mean()[0]- sample_mean[0]
```




    (19.875979962258437, 1.1325975454055133)



We can see that the sample mean differs from the population mean by 1.13 years. We can calculate a confidence interval without the population standard deviation, using the t-distribution using `stats.t.ppf(q, df)` function. This function takes in a value for the confidence level required (q) with "degrees of freedom" (df).

> In this case, the number of degrees of freedom, `df`, is equal to the sample size minus 1, or `df = sample_size - 1`. 




```python
# Calculate the t-critical value for 95% confidence level for sample taken above. 
t_critical = stats.t.ppf(q= 0.975, df= sample_size-1)   # Get the t-critical value  by using 95% confidence level and degree of freedom
print("t-critical value:")                  # Check the t-critical value
print(t_critical)     

# t-critical value:
# 2.0638985616280205
```

    t-critical value:
    2.0638985616280205
    

Calculate the confidence interval of the sample by sigma and calculating the margin of error as:
> **sigma = sample_std/√n**

> **Margin of Error = t-critical-value * sigma**

and finally the confidence interval can be calculated as : 

> **Confidence interval = (sample_mean - margin of error, sample_mean + margin of error)**


```python
# Calculate the sample standard deviation
sample_stdev = sample.std()    # Get the sample standard deviation

# Calculate sigma using the formula described above to get population standard deviation estimate
sigma =sample_stdev/ math.sqrt(sample_size)

# Calculate margin of error using t_critical and sigma
margin_of_error = t_critical* sigma[0]

# Calculate the confidence intervals using calculated margin of error 
confidence_interval = (sample_mean[0] - margin_of_error, sample_mean[0] + margin_of_error)


print("Confidence interval:")
print(confidence_interval)

# Confidence interval:
# (18.4609156900928, 21.280661568850913)
```

    Confidence interval:
    (18.56691115505689, 21.185048769459986)
    

We can verify our calculations by using the Python function `stats.t.interval()`:


```python
stats.t.interval(alpha = 0.95,              # Confidence level
                 df= 24,                    # Degrees of freedom
                 loc = sample_mean,         # Sample mean
                 scale = sigma)             # Standard deviation estimate
# (18.4609156900928, 21.280661568850913)
```




    (array([18.56691116]), array([21.18504877]))



We can see that the calculated confidence interval includes the population mean calculated above.

Let's run the code multiple times to see how often our estimated confidence interval covers the population mean value:

**Write a function using the code above that takes in sample data and returns confidence intervals**




```python
# Function to take in sample data and calculate the confidence interval
def conf_interval(sample):
    '''
    Input:  sample 
    Output: Confidence interval
    '''
    n = len(sample)
    x_hat = sample.mean()[0]
    # Calculate the z-critical value using stats.norm.ppf()
    # Note that we use stats.t.ppf with q = 0.975 to get the desired t-critical value 
    # instead of q = 0.95 because the distribution has two tails.

    t = stats.t.ppf(q= 0.975, df= n-1)  #  t-critical value for 95% confidence
    
    sigma = sample.std()/ math.sqrt(n) # Sample standard deviation

    # Calculate the margin of error using formula given above
    moe = t* sigma[0]

    # Calculate the confidence interval by applying margin of error to sample mean 
    # (mean - margin of error, mean+ margin of error)
    conf = (x_hat - moe, x_hat + moe)
    
    return conf
```

**Call the function 25 times taking different samples at each iteration and calculating the sample mean and confidence intervals**


```python
#set random seed for reproducability
np.random.seed(12)

# Select the sample size 
sample_size = 25

# Initialize lists to store interval and mean values
intervals = []
sample_means = []

# Run a for loop for sampling 25 times and calculate + store confidence interval and sample mean values in lists initialised above

for sample in range(25):

    # Take a random sample of chosen size 
    sample = pop_ages.sample(n=sample_size)
    
    # Calculate sample mean and confidence_interval
    sample_mean = sample.mean()[0]
    conf = conf_interval(sample)
   
    # Calculate and append sample means and conf intervals for each iteration
    sample_means.append(sample_mean)
    intervals.append(conf)


```

**Plot the confidence intervals along with the sample means and population mean**


```python
# Plot the confidence intervals with sample and population means
plt.figure(figsize=(15,9))

# Draw the mean and confidence interval for each sample
plt.errorbar(x=np.arange(0.1, 25, 1), 
             y=sample_means, 
             yerr=[(top-bot)/2 for top,bot in intervals],
             fmt='o')

# Draw the population mean 
plt.hlines(xmin=0, xmax=25,
           y=population_ages.mean(), 
           linewidth=2.0,
           color="red")
plt.show()
```


![png](index_files/index_20_0.png)


Just like the last lab, all but one of the 95% confidence intervals overlap the red line marking the true mean. This is to be expected: since a 95% confidence interval captures the true mean 95% of the time, we'd expect our interval to miss the true mean 5% of the time.

## Summary

In this lab, we learned how to use confidence intervals when the population standard deviation is not known, and the sample size is small (<30). We also saw how to construct them from random samples. We also learned the differences between the use cases for the $z$-score and t-distribution. We also saw how the t-value can be used to define the confidence interval based on the confidence level. 
