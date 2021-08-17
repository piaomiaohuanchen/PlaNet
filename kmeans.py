from sklearn.cluster import KMeans
import joblib
import numpy as np
class Action_trans:
    def __init__(self,filepath) -> None:
        self.K = joblib.load(filepath)
        
    def trans(self, vec_):
        vec_=vec_.reshape(1,-1)
        result = self.K.predict(vec_)
        
        return self.one_hot(result)
    
    def one_hot(self, vec_):
        targets = vec_.reshape(-1).astype(np.int32)
        result_ = np.eye(150)[targets].astype(np.int32)
        result_ = result_.reshape(-1)
        return result_