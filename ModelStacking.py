import numpy as np
import scipy
import time
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.metrics import r2_score,mean_squared_error,f1_score
from sklearn.utils import check_random_state
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Function which performs a k-fold spilt of training data (X,y) and for each 
# fold, fits the model est without the hold out set, and then makes predictions 
# on the hold out set.  This returns the predictions for each row of X using 
# this method
#
# Inputs:    est:  Estimator (should implement fit&predict)
#            X_train:    Predictors
#            y_train:    Targets
#      num_cv:     Number of folds (default=5)
#     stratify:    Boolean, use stratified k folds (labels=y_train? (Default False)
#     shuffle:     Randomly shuffle data before cv (only if stratify=False)
# predict_proba:   Boolean.  Use predict_proba method on est to generate output
#  score_func:     Scoring function to assess skill of est on folds  
#                  Format:  score_func( observation, predictions ) (default mse)
#     verbose:     Integer (0,1, or 2) indicating verbosity level.
#     x_test:      Numpy array contianing test data.  If this is provided, there
#                  will be an additional output array with size 
#                  (x_test.shape[0], y_train.shape[1], num_cv) with each 3rd
#                  dim index K is the result of applying est.predict to x_test 
#                  after est is fit using fold K.  Default: None.
#
#    Notes:  -This will not work for regression tasks with multiple outputs.
#
# Outputs:   cv_pred:   Predictions for each row of X_train. 
#                       Size: X_train.shape[0] if predict_proba is False
#                             (X_train.shape[0],num_categories) if predict_proba is True
#            cv_scores:  Array of scores computed for each fold
#            test_pred: (only if x_test != None):  Array of predictions for 
#                       all folds for x_test 
#
def cv_predict(est,X_train,y_train,num_cv=5,stratify=False,shuffle=False,
                            predict_proba=False, 
                            score_func=mean_squared_error,
                            verbose=0,x_test=None):
    # Partition training data
    if stratify:
        kf = list( StratifiedKFold(y_train,n_folds=num_cv) )
    else:
        kf = list( KFold(y_train.shape[0],num_cv,shuffle=shuffle ) )
        
    # Initialize outputs
    cv_scores = np.zeros(num_cv)
    num_classes = 1
    if not predict_proba:
        # Only one output (calling predict)
        cv_pred   = np.zeros( X_train.shape[0] )
        if x_test is not None:
            test_pred = np.zeros((x_test.shape[0],num_cv))
    else:
        # Multiple outputs (one for each class)
        le = LabelEncoder().fit(y_train)
        num_classes = len(le.classes_)
        cv_pred = np.zeros( (X_train.shape[0],num_classes) )
        if x_test is not None:
            test_pred = np.zeros((x_test.shape[0],num_classes,num_cv)) 
    
    # Train each model on cv partitions
    start_whole = time.time()
    for i,(train,test) in enumerate(kf):
        start_fold = time.time()
        x_train_fold = X_train[train]
        y_train_fold = y_train[train]
        x_test_fold  = X_train[test]
        y_test_fold  = y_train[test]    
        est.fit(x_train_fold,y_train_fold)
        if not predict_proba:
            y_pred_fold = est.predict(x_test_fold)
            cv_pred[test] = y_pred_fold
            if x_test is not None:
                test_pred[:,i] =  est.predict(x_test)
        else:
            y_pred_fold = est.predict_proba(x_test_fold)
            cv_pred[test,:] = y_pred_fold
            if x_test is not None:
                test_pred[:,i] =  est.predict_proba(x_test)
        if score_func is not None:
            cv_scores[i] = score_func(y_test_fold,y_pred_fold)
        if verbose>1:
            time_fold = time.time()-start_fold
            print "[Fold# {i}, score={score}, timing: {time} sec]".format(
                                           i=i,score = cv_scores[i],time=time_fold)
    if verbose > 0:
        time_whole = time.time() - start_whole
        print "[Done: score={score}, timing: {time} sec]".format(
                                           score = np.mean(cv_scores),time=time_whole) 
            
    if x_test is not None:
        return cv_pred,cv_scores,test_pred
    else:
        return cv_pred,cv_scores

# A class used as the default method for combining models in Stacked Regression.
# This computes non-zero weight coefficients for each stacked model.
class ConstrainedLeastSquaresCombiner(object):
    def __init__(self,objective=mean_squared_error):
        self.coef_ = None
        self.objective = objective
    
    def fit(self,X,y):
        n_cols= X.shape[1]
        beta0 = np.ones(n_cols,dtype=np.float64)/np.float64(n_cols)
        bounds = ((0,1),)
        for i in range( X.shape[1]-1):
            bounds = ((0,1),)+bounds
        self.coef_ = minimize(self.obj_fun,beta0,args=(y,X),
                              bounds=bounds,method='SLSQP').x
                             
        return self                           
    
    def predict(self,X):
        return X.dot(self.coef_)
    
    def obj_fun(self,beta,y,X):
        return self.objective(y,X.dot(beta))



# Base class used for stacked regressor and classifier classes
class StackedModel(object):
    def __init__(self,models=[],combiner=None,train_on_full=False, 
                 num_cv=5, verbose=0,score_func=None,predict_proba=False,
                 seed=None,shuffle=False,stratify=False):
        self.models=models
        self.combiner=combiner
        self.train_on_full=train_on_full
        self.num_cv = num_cv
        self.verbose = verbose
        self.predict_proba = predict_proba  
        self.score_func =score_func
        self.seed = seed
        self.shuffle=shuffle
        self.stratify=stratify
        self.test_pred = None
        
    def get_params(self,deep=True): 
        return {'models':self.models,
                 'combiner':self.combiner,
                 'train_on_full':self.train_on_full,
                 'num_cv':self.num_cv,
                 'verbose':self.verbose,
                 'predict_proba':self.predict_proba,
                 'score_func':self.score_func,
                 'seed':self.seed,
                 'shuffle':self.shuffle}
    
    # Computes model output for each model, and combines them        
    def predict(self,X):
        x_trns = self.transform(X)
        return self.combiner.predict(x_trns)
    
    # Combines output from each model in self.models
    # If x_test input was provided to fit, this uses the mean of all predictions 
    # across each fold.
    def combine(self,model_out=None):
        if model_out is None:
            return self.combiner.predict(self.test_pred)
        else:
            return self.combiner.predict(model_out) 

# Class which implements stacked regression
class StackedRegressor(StackedModel):
    def __init__(self,**kwargs):
        super(StackedRegressor,self).__init__(**kwargs)
        # Set some defaults for regression
        if self.score_func  == None:
            self.score_func = r2_score
        if self.combiner==None:   
            self.combiner = ConstrainedLeastSquaresCombiner()
           
    # Fits the stacked regression
    def fit(self,X,y,x_test=None):
        model_names = self.models.keys()
        
        # Shuffle rows?
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            v = np.random.permutation(X.shape[0])
            # Permute along first dimension
            X = X[v]
            y = y[v]
            
        # For each model, compute cross val predictions for each row of X
        stacked_train = np.zeros( (X.shape[0], len(model_names)) )
        avg_score = np.zeros(len(model_names))
        if x_test is not None:
            self.test_pred = np.zeros( (X.shape[0],len(model_names)) )
        for j,mod in enumerate(model_names):
            out = cv_predict(self.models[mod],X,y,
                             num_cv=self.num_cv,
                             stratify=False,
                             shuffle=False,
                             predict_proba=False, 
                             score_func=self.score_func,
                             verbose=self.verbose,
                             x_test=x_test)
            stacked_train[:,j] = out[0]
            avg_score[j] = np.mean(out[1])
            if x_test is not None:
                # Average over cross validations
                self.test_pred[:,j] = np.mean(out[2],axis=1)                                     
        # Print model scores
        if self.verbose>0:
            for j,mod in enumerate(model_names):
                print "[SUMMARY: {mod}, average score={score}]".format(\
                                                    mod=mod,score=avg_score[j])
        # Train each model on full datasets    
        if self.train_on_full:
            for mod in model_names:
                start = time.time()           
                self.models[mod].fit(X,y)
                if self.verbose >0:
                    print ("[Trained {mod} on full training data, timing: "
                    + "{time}]").format(mod=mod,time=time.time()-start)    
        # Combine models
        if self.combiner != None:
            if self.verbose>0:
                print "Training combiner"
            self.combiner=self.combiner.fit(stacked_train,y)
        return self
        
    def transform(self,X):
        out = np.zeros((X.shape[0],len(self.models)))
        for j,mod in enumerate(self.models):
            out[:,j] = self.models[mod].predict(X)
        return out   
    
class StackedClassifier(StackedModel):
    def __init__(self,**kwargs):
        super(StackedClassifier,self).__init__(**kwargs)
        # Set some defaults for classification
        if self.score_func==None:
            self.score_func = f1_score
        if self.combiner==None:   
            self.combiner = LogisticRegression()
    
    # Fits the stacked classifier
    def fit(self,X,y,x_test=None):
        model_names = self.models.keys()
        le = LabelEncoder().fit(y)
        self.num_classes = len(le.classes_)
        
        # Shuffle rows?
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            v = np.random.permutation(X.shape[0])
            # Permute along first dimension
            X = X[v]
            y = y[v]
            
        # For each model, compute cross val predictions for each row of X
        # Add another dimension if prediciton class probabilities
        if not self.predict_proba:
            stacked_train = np.zeros((X.shape[0], len(model_names)))
            if x_test is not None:
                self.test_pred = np.zeros( (X.shape[0],len(model_names)) )
        else:
            stacked_train = np.zeros((X.shape[0],self.num_classes*len(model_names)))
            if x_test is not None:
                self.test_pred = np.zeros( (X.shape[0],self.num_classes*len(model_names)) )
        
        avg_score = np.zeros(len(model_names)) 
        for j,mod in enumerate(model_names):
            out = cv_predict(self.models[mod],X,y,
                             num_cv=self.num_cv,
                             stratify=self.stratify,
                             shuffle=False,
                             predict_proba=self.predict_proba, 
                             score_func=self.score_func,
                             verbose=self.verbose,
                             x_test=x_test)
                             
            if not self.predict_proba:
                stacked_train[:,j] = out[0]
                if x_test is not None:
                    # Take most common vote over all folds
                    self.test_pred[:,j] = scipy.stats.mstats.mode(out[2],axis=1)
            else:
                j_start = j*self.num_classes;
                j_end = (j+1)*self.num_classes;
                stacked_train[:,j_start:j_end] = out[0] 
                if x_test is not None:
                    # Average probability over folds
                    self.test_pred[:,j_start:j_end] = np.mean(out[2],axis=2)
                
            avg_score[j] = np.mean(out[1])
                                             
        # Print model scores
        if self.verbose>0:
            for j,mod in enumerate(model_names):
                print "[SUMMARY: {mod}, average score={score}]".format(\
                                                    mod=mod,score=avg_score[j])
        # Train each model on full datasets    
        if self.train_on_full:
            for mod in model_names:
                start = time.time()           
                self.models[mod].fit(X,y)
                if self.verbose >0:
                    print ("[Trained {mod} on full training data, timing: "
                    + "{time}]").format(mod=mod,time=time.time()-start)    
        # Combine models
        if self.combiner != None:
            if self.verbose>0:
                print "Training combiner"   
            self.combiner=self.combiner.fit(stacked_train,y)
        return self

    def transform(self,X):
        if not self.predict_proba:
            out = np.zeros((X.shape[0],len(self.models)))
            for j,mod in enumerate(self.models):
                out[:,j] = self.models[mod].predict(X)
        else:
            out = np.zeros((X.shape[0],self.num_classes*len(self.models)))
            for j,mod in enumerate(self.models):
                j_start = j*self.num_classes;
                j_end = (j+1)*self.num_classes;
                out[:,j_start:j_end] = self.models[mod].predict_proba(X)
        return out 
    
    # Computes model output for each model, and combines them        
    def predict_proba(self,X):
        x_trns = self.transform(X)
        return self.combiner.predict_proba(x_trns)
    
    # Combines output from each model in self.models
    # If x_test input was provided to fit, this uses the mean of all predictions 
    # across each fold.
    def combine_proba(self,model_out=None):
        if model_out is None:
            return self.combiner.predict_proba(self.test_pred)
        else:
            return self.combiner.predict_proba(model_out)     
            
                    
