## End-to-End Machine Learning Project 
# Student Performance Prediction

## Project Description 

This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.

Here we train the Model using different features of students and predict the MATH SCORE Performance as a regression Problem.

[Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977]
-------------------------------------------------

## Getting Started
  ## Table of Contents
- [Getting Started](#getting-started)
	- [Tools Required](#tools-required)
	- [Installation](#installation)
- [Development](#development)
- [Running the App](#running-the-app)
- [Deployment](#deployment)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)
  
### Tools Required

All tools required go here. You would require the following tools to develop and run the project:

* A text editor or an IDE (like VsCode)
* Github Account [For Code Upload]
* Anaconda or Python [ For Create Virtual Environment ]

### Installation

All installation steps go here.

* Installing an Anaconda via a .exe file [Set the environment Path ](by Default it is done when installed)
  * Create a project folder
  * Open CMD and RUN -->`conda activate`
  * RUN `conda create -name newenv python=3.8 -y`
  * RUN `conda activate newenv`
  Inside the newenv you install all libraries to run the project
  after this you simply [clone the GitHub repository and Run requireiment.txt]
  * Run on the same Folder where your project requirements.txt file is available
  * Run `python install -r requirements.txt`

## Development

This section gives some insight and a basic overview of Development.

#### Life cycle of Machine Learning Project

#### Do EDA Task --> In Experiment.ipynb

- Understanding the Problem Statement
- Data Collection
- Data Checks to perform
- Exploratory data analysis
- Data Pre-Processing
- Model Training
- - Choose the best model

### 1) Problem statement
- This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.


### 2) Data Collection
- Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
- The data consists of 8 columns and 1000 rows.
  
### 2.1 Import Data and Required Packages

####  Importing Pandas, Numpy, Matplotlib, Seaborn, and Warnings Library.
- data =pd.read_csv()
- df.shape

### 2.2 Dataset information
- gender: sex of students  -> (Male/female)
- race/ethnicity: ethnicity of students -> (Group A, B, C, D, E)
- parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
- lunch: having lunch before the test (standard or free/reduced) 
- test preparation course: complete or not complete before the test
- math score
- reading score
- writing score

### 3. Data Checks to perform

- Check Missing values
- Check Duplicates
- Check data type
- Check the number of unique values in each column
- Check statistics of the data set
- Check various categories present in the different categorical column

- - Basic info 
    - `[df.shape, df.isnull().sum(), df.duplicated().sum(), df.dtypes, df.info(), df.columns,  ]`
    - [checking the count of the number of the unique values of each column --> `df.nunique()`]
    - [check stats of data -->`df.describe()`]
    - checking the get unique value  of each column --> `df.unique()`

- ### 4. Exploring Data ( Visualization ) 
    - `Matplotlib and Seaborn`[Histogram, Kernel Distribution Function (KDE), pie, bar, Boxplot(check outliers), pairplot]
    - Multivariate analysis using pieplot
    - Feature Wise Visualization
    - UNIVARIATE ANALYSIS
    - BIVARIATE ANALYSIS
 
- ### 5. MODELING
    - Importing Sklearn, Pandas, Numpy, Matplotlib, Seaborn etc.
    - Preparing X and Y variables
    - Transform the into a numerical datatype to perform Models
    - Create Column Transformer 
    - preprocessing data using OneHotEncoder, StandardScaler
    - Create an Evaluate Function to give all metrics after model Training
        - [mae, rmse, r2_square, mse] For  regression Problem
    - Create Model lists and run using a loop at once so that there no repeat same task for all model
        - ```[models ={
                    - "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                }]
                ```
    - Results [Choose the best model with the help of the evaluate Function, especially using R2Score]
        - Now predict the model `lin_model.predict(X_test)`
        - Plot y_pred and y_test [visualize data] `plt.scatter(y_test,y_pred)`
            - `sns.regplot(x=y_test,y=y_pred,ci=None,color ='red');`
            - Difference between Actual and Predicted Values 
                - `pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})`

THEN
- ### Run on Local system
- Open the  terminal
- activate the conda environment 
`conda activate newenv`
- go to the  Project directory/folder like me
- RUN `python app.py`
- Go to Browser paste localhost `http://127.0.0.1:5000/`
- Awesome Project run on your localhost
  
## License

`Student Performance Prediction [Machine Learning End-to-End Project] ` is open-source software.

## Acknowledgments

This section can also be called `Resources` or `References`

* Code Help [krish naik GitHub]
* Tutorials followed  [krish naik ml project video]


[//]: # (HyperLinks)













