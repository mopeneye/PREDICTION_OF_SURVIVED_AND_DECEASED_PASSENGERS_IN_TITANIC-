# PREDICTION_OF_SURVIVED_AND_DECEASED_PASSENGERS_IN_TITANIC

Overview
Probably all of us know that on April of 1912, Titanic sank after hitting an iceberg. Approximately 70 percent of passengers died. The aim of this project is predicting survived or deceased passengers of Titanic after disaster. In this project, I’ll use CRISP-DM (Cross-industry standard process for Data Mining project method) methodology. According to CRISP-DM there are six steps.(Business unsderstanding, Data understanding, Data preparation, Modelling, Evaluation and Deployment )
1.	Business Understanding 	
•	Determining business objectives
Traditionally, you might specify the goals, problems and resources then you should define the objective and assess current situation etc. that you would use in a CRISP-DM project you’re designing in a company, however on this specific machine learning project, objective is very precise. The goal of the project is, predicting the survived and deceased passengers of Titanic.
2.	Data Understanding 	
•	Collecting data
I’ve collected data set from http://www.kaggle.com Kaggle is a data science hosting datasets and a competition web site. As a software language, python will be used for this project. After importing some data science specific libraries and loading dataset hosted from Kaggle, we should describe data.
•	Describing data
I do not think it’s necessary to give technical details (which libraries have been used or what kind of code did I use etc.) when the extent of this assignment has been thought. I’ll try to describe or explore the amounts and types of data then I’ll use some visualisation tools of python. There’re some columns probably we’ll need like pclass (passenger class depending on their seat in the ship) , sex or gender of the passenger, age, sibsp(indicates the count of sisters or brothers or wives etc.), parch(indicates the number of parents / children), fare (how much they paid for the ticket), cabin (the cabin they are in), embarked (if it is known port of embarkation)
3.	Data Preparation 
•	Selecting, cleaning and formatting data
I used some specific visualisation tools in order to find out which data is missing. (like seaborn heat map ) In order to specify the survived and not-survived counts of passenger or with different words find out the frequency of linked variable, I’ve used matplotlib’s count plot visualisation method. None survival or deceased passenger ratio is approximately 30%. After the usage of sex column as a hue, gender of the passengers that survived or deceased may be understood. I saw that people did not survive is much more like male and approximately 65% of people that did survive was female.
I also saw that, more than half of people that could not survive was in third class passengers (this is the worst class for tickets) More or less the half of the passengers survived was high class, but expectedly 75% of third class passengers died. (3 class seats are at the bottom part of the ship and tickets are the cheapest.)
I’ve imputed missing values for age column with a function designed to fill the missing age values using their passenger class value.. 
I chose to drop the cabin column since lots of data were missing. 
Also necessary actions for the categorical columns (like sex, or embarked)  taken.
Name and ticket columns has been excluded since it is not going to be used.
4.	Modelling
Generally, more than one model should be used for a project, then we choose one of the models and it will be assessed. I’ve used logistic regression, random forest and naïve bayes ml models on this project. 
•	Building the model
At this point of model there’re 8 independent features (pclass, age, sibsp, parch, fare, male and S and L columns for the embarked column after converting it to two columns in order to numeric values. One dependent variable (survived – predicting the value of it)
I fit my train data with logistic regression model instance. 
•	Assessing the model 
I’ve performed the prediction tasks of machine learning model.  
5.	Evaluation
•	Evaluating the results
Executed the classification report and confusion matrix methods of sklearn library of python. and obtained accuracy scores below:
Logistic regression - 79%
Naïve-Bayes – 81%
Random forest -  84.29%
XGB - 84.51%
LGBM - 84.40%

6.	Deployment
According to our final report results, I’ve decided to deploy XGB ml model.

7.	Conclusion
Different feature engineering jobs might be accomplished for the next versions in order to improve the result of ml model for a couple of columns, for instance ticket, but for now it seems acceptable. 
