
# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Group Project 5: Protests - Predicting Government Response

Project Members: Stephen Strawbridge, Nicholas Kovacs, Young Park, Chris Burger
Cohort #1019

### Problem Statement
We hypothesize that protest groups are not reaching their full potential in executing successful protests.  This project aims to find the best prediction models for advising protest groups on how likely they are to receive a specific government response to their protest.

---
### Dataset Used

**Data Mobilization Research Dataset**: The dataset includes protests against governments around the world, covering 162 countries between 1990 and March 2017.  However, for our project, we only observe protests occurring after the year 2000.  Features of the dataset include protester demands, government responses, protest location, protest length, and a 'notes' column with brief descriptions of each protest.

**Data Cleaning and Modification**: In regards to null rows, null rows constituted only a very small percentage of the entire dataframe, and therefore we were confident in omitting these rows from our modeling. One crucial step in the cleaning and modification process involved dummifying the protester demand columns and the government response columns (our target variables).  Another crucial step involved applying natural language processing (NLP) and transformation to the 'notes' column in the dataframe.  Using 'featureunion', the notes column was combined with other quantitative features (such as time length and number of protesters) in the models. 

---
### Repository Table of Contents (in chronological order)

* **Notebook 1 - Protests** -  Entire main coding notebook for the project.
* **Notebook 2  - EDA** -  Further Exploratory Data Analysis.
* **data** -  Raw and cleaned protests dataset.
* **images** -  Images used for presentation.
* **model_results** -  Score results from 4 models, as well as further results on our most successful model, the XGBoost.
* **predict_protest.py** -  Predictor program
* **Project 5 slides** -  Powerpoint presentation on project.


---
### Software Requirements

* Numerical Python (numpy)
* Pandas (pd)
* Matplotlib.pylot (plt)
* Seaborn (sns)
* Scikitlearn

---

### Summary
Overall, beating the baseline scores for the 7 different responses was difficult, but was feasible in the non-violent responses.  Responses such as killings and shootings were harder to predict, as these responses are relatively rare for any region in the world.  However, when consolidating the responses into 4 distinct categories, baseline scores were lower and therefore the relative success of our models was greater.  Specifically, the logistic regression and XGBoost proved to be the most successful models:  Logistic had a mean testing score of 88.28% among all regions, while XGBoost had a mean testing score of 89.15% for all regions.  Comparatively, our baseline scores for predicting responses among 4 responses was about 74.5%, which indicates our models performed well above the baseline.  In conclusion, protest groups can, with confidence, gain further assurance in how governments will respond to their protests using our modeling process.


<img src="https://cf-images.us-east-1.prod.boltdns.net/v1/static/5615998031001/37c9803b-3034-4149-9e54-2c428bb6ac25/f2ee3a67-50d7-4d4b-a971-0ac9c629ddf3/1280x720/match/image.jpg"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
---
