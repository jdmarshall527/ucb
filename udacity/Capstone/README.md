# Udacity Machine Learning Nanodegree Capstone Project

**Using Machine Learning Techniques to Project COVID-19 Contagion**

I am providing a high level summary of the project design in this readme.  For more detailed information, I have uploaded my project proposal, my project report,and my python notebook I used through Amazon's cloud computing platform.

**Data**

In this project, I downloaded csv files from Kaggle that contained the number of confirmed cases across the world: 

https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset

This data was originally tabulated by Johns Hopkins and provided on their website:

https://coronavirus.jhu.edu/

For reproducibility purposes, I have used the data available to me at this time, which spans from January 22,2020 until July 29, 2020.


After cleaning and preparing the data, I used Amazon Sagemaker's DeepAR to make projections for up to four days out for more than 250 distinct regions.
To evaluate DeepAR's efficacy, I compared this estimator to two benchmark models: a "persistence" model and a linear regression.  The persistence model hypothesizes that today's observations are an appropriate estimate for tomorrow's observation.  For the lunear regression, I structured it such that the expectation of tomorrow's observation is a linear combination of the prior day's value along with a unique binary variable encoding for each unique region.

**Files**

- Capstone proposal - my original attempt at projecting these time series.
- Capstone report - a write up of my project's methodologies and results.
- Capstone final project - my code contained in a single python notebook.

Here is the link for my proposal review:

https://review.udacity.com/?utm_campaign=ret_000_auto_ndxxx_submission-reviewed&utm_source=blueshift&utm_medium=email&utm_content=reviewsapp-submission-reviewed&bsft_clkid=3871c483-7da4-4e54-816a-9e7497a03196&bsft_uid=52ffaf2f-4192-444c-b658-e9c5d780adec&bsft_mid=022b8772-0631-4d95-82c6-069f43bae583&bsft_eid=6f154690-7543-4582-9be7-e397af208dbd&bsft_txnid=20349de9-e9a5-464c-b470-7b0558f60cd9&bsft_mime_type=html&bsft_ek=2020-07-15T15%3A13%3A44Z#!/reviews/2429090

**Python Libraries Needed**

- math
- numpy
- pandas
- matplotlib
    - pyplot 
    - ticker
- sklearn
    - model_selection.train_test_split
    - linear_model.LinearRegression
- statsmodels.tsa.arima_model.ARIMA
- six
- random
- json 
- os 
- datetime
- boto3
- sagemaker
    - get_execution_role
    - amazon.amazon_estimator.get_image_uri
    - estimator.Estimator
    - tuner
        - IntegerParameter
        - ContinuousParameter
        - HyperparameterTuner
- IPython.display
    - display
    - HTML
