Programming languages used: Python

Required packages to run the code: CSV, Numpy, Pandas, Matplotlib, SKLearn, XGBoost

How to run it to reproduce the result:

  1) Run the script "Code/GameTimePredictor.py" with "train.csv" and "test.csv" in the parent directory (one level above the directory with this script). Prediction results will be stored as "Code/sample_submissions.csv".
  
  You can specify the following arguments:
  
  i- Which model to base the predictions upon?
    
     Choices:
    
     "xgb" - XGBoost - this is also the default choice in the script (used if the user does not specify any parameter).
     "knn" - K Nearest Neigbors
     "rfr" - Random Forest
     "sgd" - Stochastic Gradient Descent
    
  ii- Whether to show training results at runtime:
   
      If you specify any value for the second parameter, the script will show you training results.
   
  Usage examples:
  
  "python Code/GameTimePredictor.py" runs the script with XGBoost model and in the mode that does not show training results.
  
  "python Code/GameTimePredictor.py xgb" runs the script with XGBoost model and in the mode that does not show training results.
  
  "python Code/GameTimePredictor.py knn" runs the script with K Nearest Neigbors model and in the mode that does not show training results.
  
  "python Code/GameTimePredictor.py rfr True" runs the script with Random Forest Regression model and in the mode that does  show training results.
  

  2) The Python notebook "Code/GameTimePredictor.ipynb" contains the same code structure as the .py script mentioned above.
  
  3) I tried a few other models (Feed Forward Neural Nets) and implementations that were specific to some of the above-mentioned models (Random Forest Regression). They are not used towards the final submission and are stored under "Code/Scratch/"
  
  
