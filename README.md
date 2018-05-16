
# Data Science Hackathon: Kickstarter

### The following content is the final presentation for the Data Science Hackathon in Hong Kong. The dataset contains details of the kickstarter project ranging from 2009 to 2017.

### Kickstarter Distribution in the Globe (2009 - 2017)

![image.png](attachment:image.png)

### Kickstarter Goal & Pledged Amount Trend in the Globe (2009 - 2016, 2017)

![image.png](attachment:image.png)

## Kickstarter Goal & Pledged Amount Trend in the Globe (2009 - 2017)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

## Logistic Regression

* __Target__: successful or not

* __Features__: category, country, goal (USD), length of campaign

![image.png](attachment:image.png)

## Decision Tree
* __Target__: successful or not
* __Features__: Time to get funded, goal real (USD)
* __Methodology__: Split data into train (75%) and test (25%) set. Apply 4 layer depth. 
* __Accuracy Score__: 66% 

![image.png](attachment:image.png)

## Light GBM
* __Target__: successful or not
* __Features__: category, main_category, currency, country, goal (USD), length of campaign, deadline month, deadline day, launch month, launch day
* __Methodology__: Split data prior 2017 into train (70%) and test (30%)  for modeling using LightGBM with a further random split (LightGBM uses cross validation in model development)

![image.png](attachment:image.png)

### Feature Importance
![image.png](attachment:image.png)

### Model accuracy by iteration
After first few iterations, cross validation results do not improve

![image.png](attachment:image.png)

## CatBoost


![image.png](attachment:image.png)
