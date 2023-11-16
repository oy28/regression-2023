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


class FCLayer:
    def __init__(self,P,Q):
        self.W=np.random.normal(loc=0,scale=np.sqrt(2/Q),size=(P,Q))

    def forward(self,x):
        self.x=x.copy()
        return self.W@x+self.b
    
    def backward(self,dL_dy,learning_rate):
        dL_dW = dL_dy @self.x.T
        dL_db = dL_dy
        dL_dX = self.W.T @ dL_dy
        self.W-= learning_rate*dL_dW
        self.b-= learning_rate*dL_db
        return dL_dX


class ReLULayer:
    def forward(self,x):
        return np.where(x > 0, x, 0)
    
    def backward(self,dL_dy, learning_rate):
        return np.where(self.x>0,dL_dy,0.0)

class NNRegressor:
    def __init__(self,epoch,n_layer,center,learning_rate):
        """_summary_

        Args:
            epoch (int): 学習回数
            n_layer (int): 層の数
            center (int): 中間層の値
            learning_rate (double): 学習率
        """
        self.epoch=epoch
        self.n_layer=n_layer
        self.center=center
        self.lerning_rate=learning_rate
        self.layers=[]
        for i in range(self.n_rayer):
            if i==0:
                P=1
            else:
                P=self.center
            
            if i==self.n_layer-1:
                Q=1
            else:
                Q=self.center

            self.layers.append(FCLayer(P,Q))
            if i<self.n_layer-1:
                self.layers.append(ReLULayer())


    def fit(self,x_sample:np.ndarray,y_sample:np.ndarray):
        for _ in range(self.epoch):
            for x,y in zip(x_sample,y_sample):
                v=x
                for layer in self.layers:
                    v=layer.forward(v)
                y_pred=np.squeeze()
                dL_dy=y_pred-y
                for layer in reversed(self.layers):
                    dL_dy=layer.backward(dL_dy,self.learning_rate)

    def predict(self,x):
        v = np.ones((1,1))*x
        for layer in self.layers:
            v=layer.forward(v)
        y_pred = np.squeeze(v)
        return y_pred
    
def build_regressor(name,kwargs_all):
    REGRESSORS =dict(
        poly=PolyRegressor,
        gp = GPRegressor,
        nn=NNRegressor,
    )
    regressor_cls =REGRESSORS[name]
    kwargs=kwargs_all[name]
    return regressor_cls(**kwargs)