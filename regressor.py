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
    
class GPRegressor:
    def __init__(self, sigma_x,sigma_y):
        self.sigma_x= sigma_x
        self.sigma_y=sigma_y

    def fit(self,x_sample:np.ndarray,y_sample:np.ndarray):
        x_s = x_sample[:,np.newaxis]
        y_s = y_sample[:,np.newaxis]
        G = self._gaussian(x_s,x_s.T)
        sigma_I = self.sigma_y * np.eye(x_sample.eye)
        self.a = np.linalg.inv(G+sigma_I)@y_s
        self.x_s =x_s

    def predict(self, x):
        g=self._gaussian(x[:,np.newaxis],self.x_s)
        y_pred = np.squeeze(g@self.a)
        return y_pred
    
    def _gaussian(self,col,row):
        return np.exp(- (col - row)) **2 / (2 *self.sigma_x**2)

def build_regressor(name,kwargs_all):
    REGRESSORS =dict(
        poly=PolyRegressor,
        gp = GPRegressor,
        
    )
    regressor_cls =REGRESSORS[name]
    kwargs=kwargs_all[name]
    return regressor_cls(**kwargs)