import StackedRegressor
import numpy as np
import sklearn.datasets
from sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score
reload(StackedRegressor)


def scoring(est,X,y):
    return r2_score(y,est.predict(X))

def main():
    
    #d = load_diabetes()
    d = sklearn.datasets.load_boston()
    X = d.data
    y = d.target
    
    models = { \
         #'Ridge':Ridge(),
         'Lasso':Lasso(),
         'GBR':GradientBoostingRegressor(),
         'RFR':RandomForestRegressor()
         }   
    sr = StackedRegressor.StackedRegressor(models=models,train_on_full=True,shuffle=True)
    
    print "Avg Random Forest Score:",np.mean(cross_val_score(RandomForestRegressor(),X,y,scoring=scoring))
    print "Avg GBR Score:",np.mean(cross_val_score(GradientBoostingRegressor(),X,y,scoring=scoring))
    print "Avg Ridge Score:",np.mean(cross_val_score(Ridge(),X,y,scoring=scoring))
    print "Avg Lasso Score:",np.mean(cross_val_score(Lasso(),X,y,scoring=scoring))
    print "Avg Stacked Regression Score:",np.mean(cross_val_score(sr,X,y,scoring=scoring))
    
if __name__ == '__main__':
    main()    

