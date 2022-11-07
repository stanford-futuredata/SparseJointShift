from sees import ShiftEstimator
import cvxpy as cp
import numpy as np 
from pykliep import DensityRatioEstimator

import seaborn as sns; sns.set()
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy

class Bbse(ShiftEstimator):
    def __init__(self):
        return 
    def _estimate_shift(self,
					   source=None,
                       target=None,
                       source_yhat=None,
                       target_yhat=None,
                       source_proba = None,
                       target_proba = None,
                       source_y = None,
					   verbose=False
					   ):
#        flat = (source[:,-1].flatten()).tolist()[0]
		
        flat = (source_y.flatten()).tolist()[0]
        #print("line 360", (source[:,-1].flatten()).tolist()[0])
        a_set = set(flat).union()
        y_size = len(a_set)
        #print("line 364", y_size)
        
        flat2 = (source_yhat.flatten()).tolist()
        
        a_set = set(flat2)
        yhat_size = len(a_set)

    
        #flat3 = (source[:,index].flatten()).tolist()[0]
        
        
        
        # a_set = set(flat3).union((target[:,index].flatten()).tolist()[0])
        # xi_size = len(a_set)

        


        # Construct the optimization problem
        
        obj = 0
        
        # Initialize the parameters
        
        r_xy = cp.Variable((y_size))

        #p_s_xy = np.zeros((xi_size,y_size))
        
        p_t_yhat = np.zeros(yhat_size)
        
        p_s_y_yhat = np.zeros((y_size, yhat_size))
        
        
        # Conditional prob makes it more accurate
        
        #p_s_x = np.zeros(xi_size)
        
        #p_t_x = np.zeros(xi_size)
        
            
        for i in range(len(source)):
            y = source_y[i]
            yhat = source_yhat[i]
            v2 = int(y)
            v3 = int(yhat)
            p_s_y_yhat[v2,v3] +=1/len(source)  #/p_s_x[v1]
            
        for i in range(len(target)):
            yhat = target_yhat[i]
            v2 = int(yhat)
            p_t_yhat[v2] +=1/len(target) #/p_t_x[v1]


        # Formulate the objective
        for j in range(yhat_size):
            sum1 = p_t_yhat[j] 
            for k in range(y_size):
                sum1 -= r_xy[k] * p_s_y_yhat[k,j] 
            obj+=cp.norm(sum1,2)
        
        #print("The y y hat matrix and its conditional number",
        #      p_s_y_yhat, 
        
        #      np.linalg.cond(p_s_y_yhat))
        # Constraints  
        constraints = [
            r_xy>=0,
            #cp.sum( cp.multiply(r_xy, p_s_xy) )==1,
            ]
                        
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve()
        #print(p_s_x_y_yhat, p_t_x_yhat, p_s_xy, r_xy.value,problem.value)
        #print("line 439", p_s_xy)
        #print("line 440", p_t_x_yhat)
        #print("line 441",p_s_x_y_yhat)
        #print("line 457", p_t_x/p_s_x)
        
        self.Est_Shift = r_xy.value, problem.value
        return r_xy.value, problem.value

    def predict_weights(self, \
                        x_samples=None,
                        y_samples=None,):
        source = x_samples
        Est_Shift = self.Est_Shift        
        weights = list()
        for i in range(len(source)):
            y = y_samples[i]
            #y = source[i,-1]
            #yhat = source_yhat[i]
                        
            if(1):
                ratio = float(Est_Shift[0][int(y)])  
            weights.append(ratio)
        return weights
      
class Klieps(ShiftEstimator):
    def __init__(self):
        return 
    def _estimate_shift(self,
					   source=None,
                       target=None,
                       source_yhat=None,
                       target_yhat=None,
                       source_proba = None,
                       target_proba = None,
                       source_y = None,
					   verbose=False
					   ):
        xtrain = source
        xtest = target
        xtrain=np.asarray(xtrain)
        xtest = np.asarray(xtest)
        kliep = DensityRatioEstimator()
        #print("klieps shape",xtrain.shape,xtest.shape,type(xtrain),type(xtest))

        kliep.fit(xtrain, xtest) # keyword arguments are X_train and X_test
        weights = kliep.predict(xtrain)
        #print("weight is",weights)
        #print("acc is", sum(acc)/len(acc))	
        result = kliep	
        self.Est_Shift = result,weights

        explain = list()
        explain.append( list(range(xtrain.shape[1])) )
        return explain

    def predict_weights(self, \
                        x_samples=None,
                        y_samples=None,):
        kliep = self.Est_Shift[0]
        weights = kliep.predict(np.asarray(x_samples))
        return weights

# Reference: https://github.com/erlendd/covariate-shift-adaption/blob/master/Supervised%20classification%20by%20covariate%20shift%20adaption.ipynb		
# This approach builds a classifer to predict whether a data point is from the source or target domain.
	
class DLCS(ShiftEstimator):
    '''
    Discriminative Learning Under Covariate Shift.
    '''
    def __init__(self):
        return 
    def _estimate_shift(self,
					   source=None,
                       target=None,
                       source_yhat=None,
                       target_yhat=None,
                       source_proba = None,
                       target_proba = None,
                       source_y = None,
					   verbose=False
					   ):
        xtrain = source
        xtest = target
        xtrain=np.asarray(xtrain)
        xtest = np.asarray(xtest)
        fulldata = numpy.concatenate((xtrain,xtest))
        labels = [1]*len(xtrain)+[0]*len(xtest)		
        labels	= numpy.asarray(labels)	 
        result = self._build_predictor(XZ=fulldata, labels=labels)
        explain = list()
        explain.append( list(range(xtrain.shape[1])) )
        return explain
    
    def predict_weights(self, \
                        x_samples=None,
                        y_samples=None,):
        DistClassifier = self.Est_Shift
        predictions_Z = DistClassifier.predict_proba(np.asarray(x_samples))
#        weights = (1./predictions_Z) - 1. 
        weights = predictions_Z[:,0]/predictions_Z[:,1] 		
        return weights
		
    def _build_predictor(self,
						 XZ,
						 labels):
        clf = RFC(max_depth=2)
        # because we can see a learn divide in the above data we 
        # could simply use logistic regression here.
        # clf = LR()

        predictions = np.zeros(labels.shape)
        skf = SKF(n_splits=2, shuffle=True, random_state=1234)
        for fold, (train_idx, test_idx) in enumerate(skf.split(XZ, labels)):
            #print ('Training discriminator model for fold {}'.format(fold))
            X_train, X_test = XZ[train_idx], XZ[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            predictions[test_idx] = probs
            self.Est_Shift = clf			
        return
