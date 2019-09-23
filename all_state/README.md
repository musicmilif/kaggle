## [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity)
#### Each row from dataset represents an insurance claim, using features to estimate "loss" column. 
230th of 3055 ( Top 8% )  
Public LB score: 1102.85291  
Private LB score: 1114.71075

#### Data 
* Training data: 188318 rows, 131 features  
* Testing data:  125546 rows, 131 features  

#### Challenge
* Meaning of features are unknown, all we know is 116 categorical features and 14 numerical features.
* Most model using MSE as loss function to optimize, but this competition use MAE as loss function.

#### Feature Engineering
* Box-Cox Transformation (log)
* Linear shifting

#### Models
* Xgboost
	* public LB score: 1108.72665
	* private LB score: 1122.17933
	* parameters tuning from Bayesian Optimization
	* 10 Folds cross validation then uniform blending
* Neural Networks
	* public LB score: 1112.86448
	* private LB score: 1123.38230
	* dimensions: 1190 - 400 - 200 - 50 - 1
	* drop out rate: 0.4 - 0.2 - 0.2
	* All activate funciton in each hidden layer are PReLU
* Linear blending