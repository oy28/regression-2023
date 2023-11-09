from matplotlib.figure import Figure
import japanize_matplotlib as _
import numpy as np

class PolyRegressor:
    def __init__(self,d):
        self.d =d
        self.p =np.arange(d+1)[np.newaxis, :]
    
    def fit(self,x_sample,y_sample):
        X_sample = x_sample[:, np.newaxis] ** self.p
        sample_XX_inv = np.linalg.inv(X_sample.T @ X_sample)
        self.a = sample_XX_inv @ X_sample.T @ y_sample[:, np.newaxis]
    
    def predict(self,x):
        X = x[:,np.newaxis] ** self.p
        y_pred=np.squeeze(X @ self.a)
        return y_pred

def build_regressor(name,kwargs_all):
    REGRESSORS =dict(
        poly=PolyRegressor,
        
    )
    regressor_cls =REGRESSORS[name]
    kwargs=kwargs_all[name]
    return regressor_cls(**kwargs)