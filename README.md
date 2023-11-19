# experimental-speed-dating-analysis
Given the topic of dating, what factors contribute the most to attract your date? You may think that attractiveness is obviously a key. However, a good sense of humour can also score you 1 in this game!

## Data Introduction:

The dataset used in this project was gathered from participants in experimental speed dating sessions performed between 2002 and 2004. During these events, guests had a four-minute "first date" with each member of the opposing gender. When the timer ran out, participants were asked if they wanted to see their date again and were asked to rate them based on six criteria: appearance, sincerity, intellect, fun, ambition, and common interests. The dataset also includes online questionnaire data collected from participants at various stages of the procedure. At the end, there are 8,378 observations with 123 variables. The main variable of interest is the Yes/No decision of each participant with respect to each of their partners, which is denoted by match. The first few lines of the dataset are shown in Figure 1.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/acc46793-993a-41e4-9a01-853f3e4d0fb0)

With this large-size dataset, my main objective is to figure out which variables do matter regarding predicting match and build a binary classification model to predict and classify the individuals in the data into two groups based on those variables.

## Data Pre-processing:
The has_null columns indicates whether the observation has missing value at any columns or not, if there is one missing value in a particular column, then has_null equals to 1. By using this variable, I recognize that there are 7,330 over 8,378 observations that have a missing value of at least one column. Therefore, using has_null column to remove missing data is not a good approach. Therefore, I checked the number of missing values of each column and found out that these following variables have relatively high missing data.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/001e99c6-dcfb-4b90-aeb7-ced153ec3269)

From this, there is no way else rather than removing these variables, so we can minimize the loss of observations due to missing values. Moreover, there are some variables with a prefix “d_” that categorizes their same variables into ranges. For example, when a person thinks that the level of importance when a partner is of same race is 5 (importance_same_race = 5), d_importance_same_race will record value of [2-5]. Generally, the "d_x" variable does not provide any additional information or insights that are not already captured by the "x" column. Therefore, I will remove all the “d_x” variables and at this point, we have 62 variables in our dataset, which match will be my outcome variable and the remaining will be the potential predictor variables.

In the current dataset, there are 58 numerical variables and 4 categorical variables, but some are automatically imported as character variables in R. Therefore, I need to convert them to the right type to run analysis. Finally, after removing missing values, the final version has 7,079 observations with 62 variables. To understand these variables, I created different types of visualizations as in Figure 2. We can see that this dataset has a relatively similar proportion of female and male groups. There is an indifferent trend in the race of each participant and their partners. In terms of their fields of study, there are 260 different fields in total. In the pie chart below, I only visualized the top 8 fields to observe what fields are mostly popular, and we can see that Business is the top one with 20.62% in this chart. I will keep only the top 6 levels: Business, MBA, Law, Social Work, International Affairs, and Electrical Engineering for further analysis with techniques like decision trees.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/5825b5a8-72e4-454d-ba65-e76a70bbe177)

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/ada82a9b-3660-4a0a-8ef7-d4be900e886a)

In this dataset, the 6 variables that are most important that are examined to measure how each participant matched or not matched their partner(s) are: Attractiveness, Sincerity, Intelligence, Fun, Ambition, and Shared Interest. However, during the processing stage, we exclude variables related to Shared Interest and Ambition attributes. Therefore, Figure 3 shows histograms for the rate of the remaining 4 attributes that each participant rated themselves and their partners. In the criteria that they rated themselves, all the attributes except sincerity have the highest value of 8 while the highest value on sincerity is 9. On the other hand, there is a tendency in rating their partners lower regarding 4 attributes.

## Data Sampling
To perform analysis on this dataset, I started with sampling methods like random sampling and cross-validation to select the “best” subset to make accurate and reliable prediction about the data population. The way I defined “best” subset here is one that has relatively low bias and low variance with high accuracy. Let’s start with random sampling by splitting the original data randomly into 2 subsets: 80% for the training set and 20% for the validation set. I then performed 3 different models such as Logistic Regression, Linear Discriminant Analysis (LDA), and Quadratic Discriminant Analysis (QDA). Figure 4 shows the ROC curves for three models, which maximize the true positives and in turn minimize the false positives. As we can see, Logistic regression model has the greatest AUC value (the area under ROC curve) so it works better for this dataset. Therefore, I decided to go ahead using Logistic Regression from now on. 

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/d2088c9f-8c04-4c08-9071-2f8ec4b9a539)

I then created a curve in Figure 5 to show the relative value of False Negative Rate for each threshold value. There is an increasing trend for FNR values as the threshold gets higher. When threshold passes 0.4, the FNR values get higher in a diminishing trend. Therefore, I selected a threshold value of 0.4 to test it out. After performing the full logistic regression model, I got the accuracy value of 85.24% as shown in Table 2.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/264e1487-387b-4533-a433-26091a32f7fd)

The next two sampling methods I used are Leave-One-Out Cross-Validation (LOOCV) and K-folds (5 and 10 folds). Using the logistic regression model, I got the accuracy value of 84.12% with LOOCV, 84.07% for 5-fold sampling, and 84.19% for 10-fold sampling. These three methods result in slightly similar accuracy, but smaller than the one we got from random sampling. Therefore, I decided to split data with random sampling method.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/f40210b9-8ce7-48d6-a319-6670e5a049a9)

## Resampling method:
Examining the distribution of match in the training set, I realized that the dataset is imbalanced. Table 3 shows us the majority group “Not matching” is about 82.4%.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/01b73d02-78c4-4fde-9798-6a5ddb412efe)

When performing data over-sampling and under-sampling, I only got 68.79% and 68.64% of accuracy respectively. I also adjust class weights to improve model performance. The idea is to assign higher weights to minority class examples and lower weights to majority class examples during model training. However, the accuracy is still low, which is only 68.08%. Therefore, we need to do variable selection. 

## Variable Selection:
First, using regsubsets function in R to get the best subset of predictors is not ideal here due to a high computational complexity cost. With forward selection and backward selection methods, the training set can be modeled with different subsets of predictors, and the optimal model has the smallest AIC value. The forward selection produces the best model with 26 variables and an AIC value of 3,912.81 while the backward selection produces one with an AIC value of 3,912.69 but with 28 variables. Based on the decision rule, I chose the model resulted from backward selection and got an 82.98% accurate prediction from the model. The subset of independent variables includes d_age, race, importance_same_race, importance_same_religion, pref_o_intelligence, pref_o_funny, attractive_o, funny_o, intellicence_important, funny_important, intelligence, attractive_partner, sincere_partne, funny_partner, like, guess_prob_liked, careers such as Business, Law, and Social Work and level of interests in a lot of activities such as tvsports, exercise, museums, art, clubbing, …

## Ridge and Lasso regression:
I conducted logistic regression analysis using a subset of predictors obtained from backward selection and found that the model's accuracy was 82.98%. To prevent overfitting and improve the model's accuracy, I performed lasso and ridge regression analyses using the same subset of predictors. The lasso regression analysis yielded an accuracy of 84.96% with the best lambda of 0.0004833955, which was significantly better than that of the logistic regression model. Similarly, the ridge regression analysis yielded an accuracy of 83.97% with the lambda value of 0.0318, which was also an improvement over the logistic regression model. These results suggest that the use of Lasso and Ridge regression can be an effective technique for improving the accuracy of logistic regression models, especially when dealing with a large number of predictors. Overall, the improved accuracy obtained from these analyses provides more confidence in the model's ability to accurately predict the outcome of interest.

## Decision Trees:
Regarding decision trees, let’s start by fitting a classification tree in order to predict match using all variables in the dataset. The variables that are used as internal nodes are like, funny_o, attractive_o, attractive_partner with the number of terminal nodes of 7, which means that the dataset has been divided into 7 distinct subgroups based on the predictor variables. The training error rate of this tree is 15.47%.In this model, the Residual mean deviance is 0.7201, which is not small enough to be the best fit.

According to the plot for this tree’s structure in Figure 7, for the root node, the model predicts that the majority class (class 0) has a probability of 0.83048, while the minority class (class 1) has a probability of 0.16952. The characteristics of the first split in the tree are based on the "like" variable. There are 3079 observations in the left branch of the split representing the number of observations in this node that satisfy the condition "like < 6.5", and 2584 observations in the right branch corresponding to the cases where "like < 6.5" is false. From this, we can see that the individuals in this dataset decided to match with their partners when the individual:
•	responded with an answer more than or equal to 6.5 to the question “Did you like your partner”, 
•	was rated 6.5 or above by their partner regarding their attractiveness on that date night, and
•	rated 7.5 or above for their partner’s attractiveness. 

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/9f974950-b308-4e1f-b3ea-cb6622c829fc)

Since this model is built on training set only, we need to test the model on validation set, which the model has not seen before. The confusion matrix between the predicted and actual values of match is shown in Table 4.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/02b29092-2314-43d8-bbf9-e47472d2b335)

From this table, the accuracy value is 82.8%. Now, we need to perform cross-validation on this tree to obtain the optimal level of tree complexity by observing other tree structures. Figure 8 shows the graphs of cross-validation errors in terms of number of terminal nodes of each tree (on the left) and the value of the cost-complexity parameter used (on the right). 

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/f6847d7e-5472-4655-a20a-9418a027cbe2)

From these two graphs in Figure 9, the tree with 4 and 7 terminal nodes results in the least cross-validation errors. The two models based on these structures produce the same accuracy, which is 84.8% - higher than the model we considered above.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/600ff388-39f6-4d3d-9141-a9bf02750cba)

In the Bagging method, we create multiple models that use all features of the dataset, and the final prediction is obtained by averaging the predictions of all models. The "OOB estimate of error rate" is equal to 14.74% indicating the estimated error rate of the model on the "out-of-bag" sample - the samples that were not included in each bootstrap sample used to train each tree in the method – is 14.74%. When testing the model on the validation set, the accuracy is 86.8%. With Random Forest method, the "OOB estimate of error rate” is 14.99%, the accuracy is 87.1%. From this, Random Forest has some kind of similar performance to the Bagging method but is better than the first model. 

Figure 10 shows the importance of each variable in the Random Forest model. The more important a variable, the more influence it has on the target variable match. Here I only display the top 10 variables. Mean decrease accuracy estimate measures the decrease in a model's accuracy when a given feature is removed, averaging across all features. On the other hand, Gini impurity is a measure of the degree or probability of a particular variable being wrongly classified when it is randomly chosen.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/275dd5bb-ea40-46c8-8fea-79dff90637b7)

Finally, Gradient Boosting is often the best tree-based algorithm in history. By performing boosting method on the training dataset, Figure 11 displays the Relative Influence plot which shows the percentage of times each feature is used to split the data across all the trees in the boosting model. This method produces a consistent output where it considers like, attractive_o, and funny_o as the three most important ones in our classification tree.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/68686474-55be-4a81-8ffc-c14d0fbf6311)

When testing our boosting model on the validation set, we need to pick an optimal threshold which yields highest accuracy. According to Figure 12 below, 0.4 is the optimal threshold, which is associated with 85.1% accuracy. 

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/6e440c17-ffdf-4e4d-bde6-15c9e2696b6c)

From all of the methods with the accuracy recorded in Table 5, we have Random Forest yields the highest accuracy rate (87.1%) so we should use the model to predict the target variable match.

![image](https://github.com/ghna17/experimental-speed-dating-analysis/assets/94765975/1e3a86fd-935d-4bc1-9fc3-06d78140db63)

## Support Vector Machine (SVM):
One of the classification methods that could potentially be used for analyzing the dataset is the Support Vector Machine (SVM) algorithm. SVM is a flexible and powerful algorithm that can be used for binary classification problems like the one in this study. However, due to the large number of predictor variables in the dataset, implementing an SVM model would be computationally intensive and require significant time and resources. Instead, simpler and computationally efficient methods, such as logistic regression or decision trees, can also be used for this purpose. By choosing these methods, I can still gain valuable insights into the classification of the match and not match groups without compromising the feasibility or efficiency of the analysis.

## Conclusion:
After applying various classification methods to the dataset, it can be concluded that to increase the likelihood of being matched, individuals should have a high score on attractiveness and possess a good sense of humor. However, it is important to note that while variables such as like and guess_prob_liked play a significant role in explaining match, they are designed as survey questions that ask the individual whether they like their partner(s) and whether they think their partner(s) like them. Therefore, it may be too general to classify future individuals based solely on these variables.

