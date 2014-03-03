# Class which implements stacked regression using sci-kit learn
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.utils import check_random_state
from scipy.optimize import minimize


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


class StackedRegressor():
    def __init__(self,models=[],combiner=ConstrainedLeastSquaresCombiner(),
                 train_on_full=False, num_cv=5, verbose=0,score_func=r2_score,
                 seed=None,shuffle=True):
        
        self.models=models
        self.combiner=combiner
        self.train_on_full=train_on_full
        self.num_cv = num_cv
        self.verbose = verbose  
        self.score_func = score_func
        self.seed = seed
        self.shuffle=shuffle

    def get_params(self,deep=True): 
        return {'models':self.models,
                 'combiner':self.combiner,
                 'train_on_full':self.train_on_full,
                 'num_cv':self.num_cv,
                 'verbose':self.verbose,
                 'score_func':self.score_func,
                 'seed':self.seed,
                 'shuffle':self.shuffle} 
                    
    # Fits the stacked regression
    def fit(self,X,y):
       
        model_names = self.models.keys()
        kf = list( KFold(y.shape[0],self.num_cv,
                         shuffle=self.shuffle,
                         random_state=check_random_state(self.seed)) )
        stacked_train = np.zeros( (X.shape[0], len(model_names)) )
        avg_score = np.zeros(len(model_names))
        # Train each model on cv partitions
        for j,mod in enumerate(model_names):
            if self.verbose>0:
                print j,mod
            avg_score_j = 0
            for i,(train,test) in enumerate(kf):
                if self.verbose>0:
                    print "--Fold",i
                x_train_fold = X[train]
                y_train_fold = y[train]
                x_test_fold  = X[test]
                y_test_fold  = y[test]    
                self.models[mod]=self.models[mod].fit(x_train_fold,y_train_fold)
                y_pred_fold = self.models[mod].predict(x_test_fold)
                score = self.score_func(y_test_fold,y_pred_fold)
                avg_score_j+=score
                if self.verbose>0:
                    print "----score ",score
                stacked_train[test,j] = y_pred_fold
            avg_score[j] = avg_score_j / self.num_cv    
            if self.verbose > 0:
                print "--avg score ",avg_score[j]
            
        # Print model scores
        if self.verbose>0:
            for j,mod in enumerate(model_names):
                print mod," avg_score = ",avg_score[j]
        
        # Train each model on full datasets    
        if self.train_on_full:
            for mod in model_names:
                if self.verbose >0:
                    print "Training ",mod," on full dataset"           
                self.models[mod].fit(X,y)
                
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
        
    def predict(self,X):
        x_trns = self.transform(X)
        return self.combiner.predict(x_trns)
    
