#%%
#import libraries
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, make_scorer
from scipy.stats import zscore, loguniform

RANDOM_STATE = 42                                                           #fixed random state
#%%
def load_gist_train_data():                                                 #function to load data
    df = pd.read_csv("GIST_Train.csv")                                      #read csv file
    print(f"Data: {df.shape}, Classes: {df['label'].value_counts()}")       #print data shape and class distribution
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number])       #select numeric features 
    y = df["label"].map({"GIST": 1, "non-GIST": 0}).astype(int).values      #encode labels as 1 and 0                                 
    return X, y                                                             #return features and labels
#%%
def preprocessing(X):                                                       #function for preprocessing                                          
    imputer = SimpleImputer(strategy="median")                              #impute missing value with median                     
    X_imputed = imputer.fit_transform(X)                                    #fit imputer and transform data            
    print("Outlier removal...")                                             #print message "outliers removal"                                        
    n_features_outlier = 0                                                  #counter for features with outliers                                                                                                                       
    n_outliers_total = 0                                                    #counter for total outliers
    X_out = np.copy(X_imputed)                                              #copy of imputed data for outlier removal
    for idx in range(X.shape[1]):                                             #iterate over features
        mean = np.mean(X_imputed[:, idx])                                        #mean of feature idx
        std = np.std(X_imputed[:, idx])                                         #std of feature idx
        mask = (X_imputed[:, idx] < mean - 3*std) | (X_imputed[:, idx] > mean + 3*std)    #mask for outliers (3 std dev from mean)
        n_outliers_idx = mask.sum()                                           #number of outliers in feature idx
        if n_outliers_idx > 0:                                                #if outliers found, update counters and set outliers to NaN for imputation
            n_features_outlier += 1                                         #increment feature outlier counter
            n_outliers_total += n_outliers_idx                                #increment total outlier counter
            X_out[mask, :] = np.nan                                         #set outliers to NaN for imputation
    X_clean = SimpleImputer(strategy="median").fit_transform(X_out)         #impute outliers with median
    print(f"Outliers: {n_outliers_total}")                                  
    scaler = RobustScaler()                                                 #robust scaler to reduce outlier influence
    X_scaled = scaler.fit_transform(X_clean)                                #scale data
    z = pd.DataFrame(zscore(X, nan_policy='omit'))                          #z-score for outlier diagnostics
    perc_out = (np.abs(z) > 3).sum() / len(X) * 100                         #percentage of outliers per feature
    print("Outliers % top10:")                                              
    print(pd.DataFrame({'perc': perc_out}).sort_values('perc', ascending=False).head(10))       #print top 10 features with most skewness              
    skew = 3 * (X.mean() - X.median()) / X.std()                                                #skewness calculation (Pearson's  coefficient of skewness)
    print("Skewness top10:")                                                                    
    print(pd.DataFrame({'skew': skew}).sort_values('skew', ascending=False, key=abs).head(10))  #print top 10 features with most skewness (absolute value)
    return X_scaled                                                                             #return preprocessed data        
#%%
def rfecv_feature_select(X, y, output_dir):                                 #function for feature selection using RFECV
    scaler = StandardScaler()                                               #standard scaler for RFECV (SVM is sensitive to feature scales)
    X_s = scaler.fit_transform(X)                                           #scale data for RFECV
    svc = SVC(kernel='linear')                                              #SVM with linear kernel for feature selection (weights can be interpreted for importance)
    rfecv = RFECV(svc, step=1, cv=StratifiedKFold(5), scoring='roc_auc')    #RFECV with SVM, 5-fold CV, and AUC as scoring metric
    rfecv.fit(X_s, y)                                                       #fit RFECV to data
    X_selected = rfecv.transform(X_s)                                            #transform data to selected features
    print(f"Features: {X.shape[1]} -> {X_selected.shape[1]}")                    #print number of features before and after selection
    plt.figure(figsize=(10,6))                                              #plot RFECV performance vs number of features
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score'])+1), rfecv.cv_results_['mean_test_score'])   #plot mean test score (AUC) vs number of features
    plt.title('RFECV') 
    plt.savefig(os.path.join(output_dir, 'rfecv.png'))
    plt.close()  
    return X_selected                                                            #return selected features after RFECV
#%% 
def grid_tune_metrics(X_selected, y, cv):                                        #function for grid search tuning and metrics evaluation
    param_distributions = {                                                 #hyperparameter grids for each classifier
        "LogisticRegression": [                             
            {
                "penalty": ["l1", "l2"],                    
                "C": [0.001, 0.01, 0.1],                     
                "solver": ["liblinear"],                    
                "class_weight": [None, "balanced"],        
            }
        ],
        "SVM": [
            {
                "kernel": ["linear"],                       
                "C": [0.001, 0.01, 0.1, 1],                
                "class_weight": [None],                      
            },
            {
                "kernel": ["rbf"],                          
            "C": [0.01, 0.1, 1, 10],                        
                "gamma": [0.001, 0.01, 0.1],                 
                "class_weight": [None],                    
            },
        ],
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15],       
            "weights": ["uniform", "distance"],          
            "p": [1, 2],                                 
        },
        "DecisionTree": {
            "criterion": ["gini", "entropy"],             
            "max_depth": [5, 10, 15, 20, 30],              
            "min_samples_split": [2, 3, 4, 5, 10],        
            "min_samples_leaf": [1, 2, 5, 7],               
            "splitter": ["random"],                          
            "ccp_alpha": [0.0005, 0.001, 0.005, 0.01],      
            "class_weight": [None],                         
        },
        "RandomForest": {
            "n_estimators": [200, 250, 300, 350],          
            "max_depth": [None, 5, 10, 15, 20],            
            "min_samples_split": [2, 5, 10, 20],            
            "min_samples_leaf": [1, 2, 4],                 
            "max_features": ["sqrt", "log2", None],      
            "bootstrap": [True, False],                    
            "class_weight": [None],                       
        },
    }
    classifiers = {                                                #base classifiers for each model type
        'LogisticRegression': LogisticRegression(max_iter=2000, solver='liblinear', random_state=RANDOM_STATE), 
        'SVM': SVC(random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
    }
    results = []                                            #list to store results for each classifier
    tuned = {}                                              #dictionary to store best tuned classifiers
    print("\n=== GRIDSEARCH + METRICS ===")
    spec_scorer = make_scorer(recall_score, pos_label=0)    #custom scorer for specificity
    for name, classifier_base in classifiers.items():                     #iterate over classifiers
        print(f"{name}...")
        search = GridSearchCV(classifier_base, param_distributions[name], scoring='roc_auc', n_jobs=-1, cv=cv) #grid search with AUC as scoring metric
        search.fit(X_selected, y)                                #fit grid search to data
        best_classifier = search.best_estimator_                   #get best estimator from grid search
        tuned[name] = best_classifier                              #store best classifier in tuned dictionary
        acc = cross_val_score(best_classifier, X_selected, y, cv=cv, scoring='accuracy')    #cross-validated accuracy
        recall = cross_val_score(best_classifier, X_selected, y, cv=cv, scoring='recall')   #cross-validated recall (sensitivity)
        auc_ = cross_val_score(best_classifier, X_selected, y, cv=cv, scoring='roc_auc')    #cross-validated AUC
        f1 = cross_val_score(best_classifier, X_selected, y, cv=cv, scoring='f1')           #cross-validated F1 score
        spec = cross_val_score(best_classifier, X_selected, y, cv=cv, scoring=spec_scorer)  #cross-validated specificity
        row = {                                                                 #dictionary to store results (AUC, recall, F1, spec) for current classifier 
            'classifier': name, 
            'tune_auc': search.best_score_,
            'auc_mean': auc_.mean(), 'auc_std': auc_.std(),
            'acc_mean': acc.mean(), 'acc_std': acc.std(),
            'recall_mean': recall.mean(), 'recall_std': recall.std(),
            'f1_mean': f1.mean(), 'f1_std': f1.std(),
            'spec_mean': spec.mean(), 'spec_std': spec.std(),
            'best_params': str(search.best_params_)
        }
        results.append(row)                                                     #append results for current classifier to results list
        print(f"  Best params: {search.best_params_}")
        print(f"  AUC CV: {auc_.mean():.4f} ± {auc_.std():.4f}")
        print(f"  Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
        print(f"  Recall (Sensitivity): {recall.mean():.4f} ± {recall.std():.4f}")
        print(f"  F1: {f1.mean():.4f} ± {f1.std():.4f}")
        print(f"  Specificity: {spec.mean():.4f} ± {spec.std():.4f}")
    pd.DataFrame(results).to_csv('grid_tuning_metrics.csv', index=False)
    return tuned

def main():                                                                     #main function to execute the workflow
    os.makedirs('results_grid', exist_ok=True)                                  #create directory for results if it doesn't exist
    X, y = load_gist_train_data()                                               #load data
    X_scaled = preprocessing(X)                                                 #preprocess data (imputation, outlier removal, scaling)
    X_selected = rfecv_feature_select(X_scaled, y, 'results_grid')                   #feature selection using RFECV
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)            #stratified 5-fold cross-validation setup
    tuned = grid_tune_metrics(X_selected, y, cv)                                     #grid search tuning and metrics evaluation for each classifier
    print("\nTuning complete. Full metrics printed + grid_tuning_metrics.csv")  

if __name__ == "__main__": 
    main()
