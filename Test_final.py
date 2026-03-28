#Import libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


fit_state = {                                       #dictonary to store fitted objects and statistics
    'imputer1': SimpleImputer(strategy="median"),   #first imputation to handle missing values before outlier detection
    'imputer2': SimpleImputer(strategy="median"),   #second imputation to handle missing values after outlier handling
    'scaler': RobustScaler(),                       #scaling to reduce the influence of outliers, fitted on training data   
    'means': None,                                  #to store means of training data for outlier detection
    'stds': None                                    #to store standard deviations of training data for outlier detection
}

def preprocess(df, is_test=False):                  #preprocessing function 
    X = df.select_dtypes(np.number)                 #select only numeric columns for preprocessing    
    print(f"Preprocess {X.shape} (Test={is_test})") #print shape of data being preprocessed, with indication if it's test or training data
    
   
    if not is_test:                                     #fitting of imputer
        X_imp = fit_state['imputer1'].fit_transform(X)  #first imputation to handle missing values before outlier detection, fitted on training data
        fit_state['means'] = np.mean(X_imp, axis=0)     #store means of training data for outlier detection
        fit_state['stds'] = np.std(X_imp, axis=0)       #store standard deviations of training data for outlier detection   
    else:
        X_imp = fit_state['imputer1'].transform(X)      #first imputation on test data using fitted imputer from training data

    X_out = np.copy(X_imp)                                  #copy of imputed data to handle outliers, initialized with imputed values       
    for j in range(X.shape[1]):                             #loop through each feature to detect and handle outliers based on training data 
        m, s = fit_state['means'][j], fit_state['stds'][j]  #get mean and std of feature j from training data for outlier detection
        mask = (X_imp[:, j] < m - 3*s) | (X_imp[:, j] > m + 3*s) #create mask for outliers in feature j based on training data statistics (values outside mean ± 3*std are considered outliers)
        X_out[mask, j] = np.nan 
    
    if not is_test:                                             #second imputation and scaling, fitted on training data
        X_clean = fit_state['imputer2'].fit_transform(X_out)    #second imputation to handle missing values after outlier handling, fitted on training data
        return fit_state['scaler'].fit_transform(X_clean)       #scaling to reduce the influence of outliers, fitted on training data
    else:
        X_clean = fit_state['imputer2'].transform(X_out)        #second imputation on test data using fitted imputer from training data
        return fit_state['scaler'].transform(X_clean)           #scaling on test data using fitted scaler from training data


# 1. Training data
df_train = pd.read_csv('Data/GIST_Train.csv')
y_train = df_train['label'].map({'GIST':1, 'non-GIST':0}).values
X_train = preprocess(df_train, is_test=False)

# 2. Test data
df_test = pd.read_csv('Data/GIST_Test.csv')
X_test = preprocess(df_test, is_test=True)

<<<<<<< HEAD
# 3. Feature selection based on indices from previous grid search
feat_df = pd.read_csv('results_grid/selected_features_indices.csv') #CSV file containing indices of selected features based on previous grid search analysis, read into a DataFrame
indices = feat_df['index'].astype(int)                              #convert indices to integer type for indexing
X_train_sel = X_train[:, indices]                                   #select only the features from training data that were identified as important in the previous analysis using the indices from the CSV file
X_test_sel = X_test[:, indices]                                     #select only the features from test data that were identified as important in the previous analysis using the indices from the CSV file
=======
# 3. Feature selection based on indices from previous grid search, selecting only the features that were identified as important in the previous analysis 
feat_df = pd.read_csv('Results/selected_features_indices.csv')
indices = feat_df['index'].astype(int)
X_train_sel = X_train[:, indices]
X_test_sel = X_test[:, indices]
>>>>>>> 42173446eb9eec665e29a9b15389d5a0e600b0ac
print(f"Selected features: {X_test_sel.shape[1]}")

# 4. Model Training
model = SVC(C=0.1, kernel='linear', probability=True, random_state=42)  #initialize Support Vector Machine (SVM) classifier with specified hyperparameters 
model.fit(X_train_sel, y_train)                                         #fit the SVM model on the selected features of the training data and corresponding labels (GIST vs non-GIST)        

# 5. Predictions and probabilities on test data
pred = model.predict(X_test_sel)                                #predict class labels 
prob = model.predict_proba(X_test_sel)[:,1]                     #predict probabilities of the positive class (GIST) for the test data 
y_test = df_test['label'].map({'GIST':1, 'non-GIST':0}).values  #encode true labels of test data as binary values (1 for GIST, 0 for non-GIST) for evaluation metrics calculation

# 6. Evaluation metrics
mis = (pred != y_test).sum()                    #calculate number of misclassifications by comparing predicted labels with true labels of test data
acc = accuracy_score(y_test, pred)              #calculate accuracy of predictions
f1 = f1_score(y_test, pred)                     #calculate F1 score of predictions 
rec = recall_score(y_test, pred)                #calculate recall (sensitivity) for the positive class (GIST) 
auc = roc_auc_score(y_test, prob)               #calculate Area Under the Receiver Operating Characteristic Curve (AUC)
spec = recall_score(y_test, pred, pos_label=0)  #calculate specificity (true negative rate) for the negative class (non-GIST) 

# 7. Save predictions and probabilities to CSV file
out = pd.DataFrame({'prediction': pred, 'prob': prob, 'true': y_test})  #create a DataFrame to store predicted class labels, predicted probabilities for the positive class (GIST)
out.to_csv('predictions.csv', index=False)                              #save the DataFrame containing predictions, probabilities, and true labels to a CSV file 

print("-" * 30)
print(f"Misclassifications: {mis}")
print(f"Accuracy: {acc:.3f}")
print(f"AUC: {auc:.3f}")
print(f"Recall GIST: {rec:.3f}")
print(f"Specificity: {spec:.3f}")
print(f"F1: {f1:.3f}")
