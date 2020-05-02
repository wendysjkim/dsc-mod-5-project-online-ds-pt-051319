## Understanding Happiness Through Machine Learning

- [Data Used](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/GSS2018.dta)
- [Jupyter Notebook](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/analysis.ipynb)
- [Summary of Findings](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/presentation.pdf)

In this project, I analyzed and predicted happiness level in the U.S. through machine learning. I used the 2018 General Social Survey data, which is available [here](https://gss.norc.org).

Since 1972, the General Social Survey (GSS) gathers data on contemporary American society in order to monitor and explain trends and constants in attitudes, behaviors, and attributes. Also, since the GSS adopted questions from earlier surveys, trends can be followed for up to 80 years.


## EDA and visualizations

> In a scale of 1-3, how happy are you?

In 2018, more than half of the respondents answered that they are "pretty happy", followed by "very happy" and "not too happy".

![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/Happiness.png)

The breakdown of happiness level is fairly consistent across each gender, age groups and region. The composition varies at different income levels, with highest income level having the least percentage of not too happy responses.

> **Gender**


![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/gender.png)
![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/Legend.png)

> **Age Groups**


![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/age.png)
> **Region**


![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/region.png)

> **Income**


![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/income.png)

One interesting question asked in the survey was number of hours that a respondent watched TV *a day*. Although more than 30% of observations were missing, it is interesting to see how proprotion of 'not too happy' cateogry peaks at around 9 hours of TV watched per day, but decreases as the number of TV hours increases.

>**TV hours**


![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/tv.png)

## Feature Selection
I only considered variables that had less than 1% missing observations. The missing threshold is very strict for the following reasons:
- The target variable has a class imbalance problem, and the missing observations in the minority classes would worsen the class imbalance problem.
- Some features have missing observations by construction. More specifically, if a respondent does not have a job, he/she won't be asked the follow-up question on how many hours he/she worked in the past week. Therefore, such features with missing observations are likely to have another features with less missing observations that have already covered the same topic, but with less detailed information.
- The raw dataset has 2,348 features. Considering how most of the variables are categorical variables, the train data set would increase exponentially when the categorical variables are converted to dummy variables. Considering only features with the most comprehensive information would reduce the computation time.

Then, I identified total of 45 features to include in the model through following approaches.

- Highest correlation to the target variable
   - 20 continous variables
   - 20 categorical variables (correlation ratio calculated as weighted variance of the mean of each category divided by the variance of all samples)
   - Among these 40 variables with high correlation, I later dropped the features that are not directly related to the respondent, i.e. type of survey questionnaire, length of the interview, etc.
- Additional potentially important variables 
   - Manually identified after reviewing the data dictionary.
   - Objectively identified through vanilla models using (a) all features and (b) all features, but focusing on the two minority classes - "very happy" and "not too happy".
   
## GridSearch
After running several vanially models, I decided to go with random forest model as it had highest weighted F1 score.

![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/corr_vanilla.png)


Then, I performed multiple grid searches to refine the model. The final model's performance is as below.


![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/corr_final_cm.png)
![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/corr_final_featimp.png)



I did similar exercise but focusing on the two minority classes, as the majority ("pretty happy") class was the category that I was least interested in. For this model, XGBoost performed the best.

![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/bi_final_cm.png)
![](https://github.com/wendysjkim/dsc-mod-5-project-online-ds-pt-051319/blob/work/images/bi_final_featimp.png)

## Conclusion
The model suffered from class imbalance problem, even though I considered features with least missing observations and oversampled the train dataset using SMOTE. 

Possible next steps are:
- Analyze each feature more deeply, especially those didn't satisfy the missing observation threshold. The variables that are dependent of prevous questions could be cleaned further.
- Since the model performance increased significantly when focused only on minority classes, I could look into predicting happiness through multiple models, i.e. first predict whether or not a respondent is "pretty happy", then predict if "not too happy" or "very happy".

