# Udacity Machine Learning Nanodegree Capstone Project

**Using Machine Learning Techniques to Project COVID-19 Contagion**

I am providing a high level summary of the project design in this readme.  For more detailed information, I have uploaded my project proposal, my project report,and my python notebook I used through Amazon's cloud computing platform.

**Data**
In this project, I downloaded csv files from Kaggle that contained the number of confirmed cases across the world: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset.  
This data was originally tabulated by Johns Hopkins and provided on their website: https://coronavirus.jhu.edu/
For reproducibility, I used the data available to me, starting on January 22,2020 until July 29, 2020.

After cleaning and preparing the data, I used Amazon Sagemaker's DeepAR to make projections for up to four days out for more than 250 distinct regions.
To evaluate DeepAR's efficacy, I compared

