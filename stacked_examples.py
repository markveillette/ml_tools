import ModelStacking
import numpy as np
import sklearn.datasets 
from sklearn.linear_model import Lasso,Ridge,LogisticRegression
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,RandomForestClassifier,GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score,accuracy_score,log_loss,f1_score
reload(ModelStacking)

def scoring(est,X,y):
    return r2_score(y,est.predict(X))

def scoring_cls(est,X,y):
    return accuracy_score(y,est.predict(X))


def run_stacked_reg():
    d = sklearn.datasets.load_diabetes()
    X = d.data
    y = d.target
    
    models = { \
         'Ridge':Ridge(),
         'Lasso':Lasso(),
         'GBR':GradientBoostingRegressor(),
         'RFR':RandomForestRegressor()
         }   
    sr = ModelStacking.StackedRegressor(models=models,train_on_full=True,shuffle=True)
    
    print "Avg Random Forest Score:",np.mean(cross_val_score(RandomForestRegressor(),X,y,scoring=scoring))
    print "Avg GBR Score:",np.mean(cross_val_score(GradientBoostingRegressor(),X,y,scoring=scoring))
    print "Avg Ridge Score:",np.mean(cross_val_score(Ridge(),X,y,scoring=scoring))
    print "Avg Lasso Score:",np.mean(cross_val_score(Lasso(),X,y,scoring=scoring))
    print "Avg Stacked Regression Score:",np.mean(cross_val_score(sr,X,y,scoring=scoring))

def run_stacked_classification():
    d = sklearn.datasets.load_iris()
    X = d.data
    y = d.target
    v = np.random.permutation(X.shape[0])
    X = X[v]
    y = y[v]
    
    models = { \
         'LDA':LDA(),
         'SVM':SVC(probability=True),
         'RFC':RandomForestClassifier(),
         }
    sc = ModelStacking.StackedClassifier(models=models,
                                         num_cv=10,
                                         train_on_full=True,
                                         score_func=accuracy_score,
                                         stratify=True,
                                         predict_proba=False,
                                         verbose=0)
    
    print "Avg LogisticRegression Score:",np.mean(cross_val_score(LogisticRegression(),X,y,scoring=scoring_cls))
    print "Avg LDA Score:",np.mean(cross_val_score(LDA(),X,y,scoring=scoring_cls))
    print "Avg SVC Score:",np.mean(cross_val_score(SVC(),X,y,scoring=scoring_cls))
    print "Avg RandomForestClassifier Score:",np.mean(cross_val_score(RandomForestClassifier(),X,y,scoring=scoring_cls))
    print "Avg Stacked Classifier Score:",np.mean(cross_val_score(sc,X,y,scoring=scoring_cls))

def main():
    run_stacked_reg()
    run_stacked_classification()
    
if __name__ == '__main__':
    main()    

