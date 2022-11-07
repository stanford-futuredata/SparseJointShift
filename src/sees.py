import cvxpy as cp
import numpy as np
import itertools
import copy
import random
import util
from util import eval_performance, compute_acc
np.random.seed(123)

class ShiftEstimator(object):
    ''' base class for shift estimation. '''
    def __init__(self):
        raise NotImplementedError

    def estimate_shift(self,
                       source=None,
                       target=None,
                       source_yhat=None,
                       target_yhat=None,
                       source_proba = None,
                       target_proba = None,
                       source_y = None,
                       ):
        '''
        shift estimation method
        parameter:
            source: n by d array, source feature data
            target: m by d array, target feature data
            source_yhat: n by 1 array, predicted labels on source data 
            target_yhat: m by 1 array, predicted labels on target data
            source_proba: n by L array, predicted label prob on source data
            target_proba: m by L array, predicted label prob on source data
            
        return: 
            shiftfeature: a list of indices, indicating the shifted features
            shift: estimated performance differences between target and source.
        It equals to acc(target) - acc(source).
        '''
        result = self._estimate_shift(
                       source=source,
                       target=target,
                       source_yhat=source_yhat,
                       target_yhat=target_yhat,
                       source_proba = source_proba,
                       target_proba = target_proba,
                       source_y = source_y,
                       )
        
        weights = self.predict_weights(source,source_y)
        acc_source = [source_y[i]==source_yhat[i] for i in range(len(source))]
        acc = eval_performance(weights,acc_source)        
        return result[0], acc[0]-sum(acc_source)/len(acc_source)

    def set_params(self,
				   s = 1,
                   eta = 0.01,
                   kernel = "BF",
                   fix_index=False,
                   index_set=[6],
				   post_sparse=False,
				   verbose=False,
				   use_fix=False,
                   soft=None,
                   max_iter=20,
				   ):
        self.s = s				   
        self.eta = eta				   
        self.kernel = kernel				   
        self.fix_index = fix_index				   
        self.post_sparse = False			   
        self.index_set = index_set
        self.verbose=verbose
        self.use_fix = use_fix		
        self.indexes_learned = []	
        self.s = s				
        self.soft = soft		
        self.max_iter = max_iter
        self.verbose = verbose
        return 	
    
    def predict_weights(self, \
                        x_samples=None,
                        y_samples=None,):
        raise NotImplementedError
		
    def getindex(self,s):		
        indexes_learned =self.indexes_learned
        while(len(indexes_learned)<s):
            indexes_learned.append(-1)
        return indexes_learned			
        
class Seesc(ShiftEstimator):
    ''' The shift estimation and explanation under 
    sparse joint shift model. 
    '''

    def __init__(self):
        self.eps = 1e-10
        self.s = 1				   
        self.eta = 0.01				   
        self.kernel = "BF"	 # {Lin, quadsingle, BF}			   
        self.fix_index = False				   
        self.post_sparse = False			   
        self.index_set = [6]	
        self.verbose = True
        self.use_fix = False
        self.indexes_learned = []				
        return 
	
		
    def getindex(self,s):		
        indexes_learned =self.indexes_learned
        while(len(indexes_learned)<s):
            indexes_learned.append(-1)
        return indexes_learned			
		

    
    def _estimate_shift(self,
                       source=None,
                       target=None,
                       source_yhat=None,
                       target_yhat=None,
                       source_proba = None,
                       target_proba = None,
                       source_y = None,
                       ):
        '''
        shift estimation method
        parameter:
            source: n by d array, source feature data
            target: m by d array, target feature data
            source_yhat: n by 1 array, predicted labels on source data 
            target_yhat: m by 1 array, predicted labels on target data
            source_proba: n by L array, predicted label prob on source data
            target_proba: m by L array, predicted label prob on source data
            
        return: 
            shiftfeature: a list of indices, indicating the shifted features
            shift: estimated performance differences between target and source.
        It equals to acc(target) - acc(source).
        '''
        
        # 1. prepare the variables	
        s = self.s				   
        eta = self.eta				   
        kernel = self.kernel				   
        fix_index = self.fix_index				   
        post_sparse = self.post_sparse				   
        index_set = self.index_set	
        verbose = self.verbose
        self.fix_index = fix_index
        self.index_set = index_set
        
        y_set = np.unique(np.asarray(source_y)).flatten()
        x_set = [np.unique(np.asarray(source[:,i])).flatten() 
                 for i in range(source.shape[1])]
        #x_set = np.unique(np.asarray(source),axis=0).flatten()
        self.x_set= x_set
        xit=np.asarray(source[0,:]).flatten()
#        print("xit", xit.ravel())
        test = self._ratiofunc(x=xit,kernel=kernel)
        K = len(test)
        L = len(y_set)
        groupx = self._ratiogroup(xit,kernel=kernel)
#        print("first test and group results", test,groupx)
#        print("line 72 K and L: ",K, L)

        # 2. set up the variables
        Z = list()
        for i in range(len(y_set)):
            Z.append(cp.Variable(K))
            
        # 3. form the objective
        obj= 0
        target = np.asarray(target)
#        print("len of target",len(target),target)
        for i in range(len(target)):
            p_y = target_proba[i,:]
            xit = np.asarray(target[i,:]).flatten()
            phi = self._ratiofunc(xit,kernel=kernel)
            phi = phi.flatten()
            #print("phi is",phi)
            w1 = [Z[j]@phi for j in range(len(Z))]
            #print("line 1486", len(Z), p_y)
            #print("line 1487", Z[0][0], phi,Z[0]@phi)
        
            obj+=cp.log(w1@p_y)/len(target)
			
        # 4. form the constraints
        constraints = [
#            W>=0,
#            cp.sum( cp.multiply(W, ps_x_y))==1,
            ]
        s1 = 0
#        print("line 1497 source y", source)
        for i in range(len(source)):
            y = source_y[i]
            xit =np.asarray(source[i,:]).flatten()
            phi = self._ratiofunc(xit,kernel=kernel)
            #print("line 1501, y is",y)
#            s1+=Z[int(y)]@phi/len(source)
            p_y = source_proba[i,:]
            w1 = [Z[j]@phi for j in range(len(Z))]
            s1+=w1@p_y/len(source)				
        constraints+=[s1==1]
        
        for i in range(L):
            constraints+=[Z[i]>=0]
        constraint_noreg = copy.copy(constraints)
        # add regularizer
#        if(1):
        if(fix_index==False):
            reg = 0
            reg_var = list()
            for i in range(target.shape[1]):
                s1 = 0			
                reg1 = cp.Variable(L*len(groupx[i]))
                for j in range(L):
    #                if(not(i==4)):
    #                    constraints+= [Z[j][i*2]==0]					
    #                    constraints+= [Z[j][i*2+1]==0]			
                    K = len(groupx[i])			
                    for k in range(len(groupx[i])):
                        m = int(groupx[i][k])				
                        constraints+= [reg1[K*j+k] == Z[j][m]]
					
						
                reg_var.append(reg1)			
                reg+=cp.norm2(reg1)	
        #print("regular", reg)
        #print("constraints",constraints)
#        eta = 10 # good for covid
            objr = -obj+reg * eta	
            
        if(fix_index==True):
            eps = 1e-6
            objr = -obj #+reg * eps
            print("fix index with small eta:",eps)
			
        problem = cp.Problem(cp.Minimize(objr), constraints)
		
        #problem.solve(verbose=verbose,solver=cp.CVXOPT)  
		
        try:
            if(verbose):		
                print("try the MOSEK solver.")
            problem.solve(verbose=verbose,solver=cp.MOSEK) 
        except:
            print("MOSEK solver is NOT installed. Switch back to SCS solver.")
            problem.solve(verbose=verbose,solver=cp.SCS)	
        #print("result is", Z[0].value,Z[1].value, problem.value)
        index = index_set
        indexes = index

        if(fix_index==False):        		
            values = [np.sum(regt.value) for regt in reg_var]
            index = np.argmax(values)
            values = np.array(values)
            indexes = values.argsort()[::-1]
            index_set = []
            i = 0
            while (i<s):
                index_set.append(indexes[i])
                i+=1
            if(self.verbose==True):
                print("identified indexes (and values) are", index_set, values)
			
        #print("post sparse is",post_sparse)
        if(post_sparse==False):  		
            self.kernel = kernel
            self.index = index
            self.indexes_learned = indexes
            self.Z = Z
            subset = index
            params = list()
            for i in Z:
                params.append(i.value)		
            return subset, params

    def _transform_x(self,xit,index_set=[5]): 
        return xit[index_set]	
	
	
    def predict_weights(self, x_samples=None,
                        y_samples=None,
						):
        '''
        predict the weights for a given test sets
        '''
        use_fix = self.use_fix
#        use_fix = False		
        index_set =	self.index_set	
        eps = self.eps
        weights = list()
        for i in range(len(x_samples)):
            x = np.asarray(x_samples[i]).flatten()
            if(use_fix==True):			
                x = self._transform_x(x,index_set)
            w1 = 0
            y = int(y_samples[i])
#            for y in range(len(x_proba[i])):
            if(1):
                phi = self._ratiofunc(x,kernel=self.kernel)
                Z = self.Z
                a1 = [Z[y].value[j]*phi[j] for j in range(len(phi))]
                w1 += sum(a1)
            if(w1<0):
                w1 = eps
            weights.append(w1)
            
        return weights
        
    def _ratiofunc(self,x,kernel="Lin"):
        fix_index  = self.fix_index
        index_set = self.index_set
        if(kernel=="Lin"):
            #print(x)
            phi = np.zeros(len(x)+1)
            phi[0:len(x)] = x
            phi[-1]= 1
#            phi[6] = (x[6]==1)
        if(kernel=="quadsingle"):
            phi = np.zeros(2*len(x)+1)
            phi[0:len(x)] = x
            phi[len(x):len(x)*2] = np.multiply(x,x)
            phi[-1]= 1
            #print("x and phi",x,phi)
			
#            phi[6] = (x[6]==1)
        if(kernel=="Binary"):
            w = np.zeros(2*len(x))
            #groupw = list()
            for i in range(len(x)):
                z1 = np.zeros(2)
                z1[0] = 2*i
                z1[1] = 2*i+1	
                #groupw.append(z1)				
                w[2*i] = 1.0*(x[i]==0)
                w[2*i+1] = 1.0*(x[i]==1)
            phi = w
        if(kernel=="BFFix"):
            phi = list()	
            j = 0			
            for i in range(len(self.x_set)):
                if(not(i in index_set)):
                    continue					
                for value in self.x_set[i]:
                    if(1):
                        #print("x[j]",j,value,index_set,x)					
                        phi.append(1.0*(x[j]==value))
                j += 1 						
            #phi.append(1) # disable this if a constant is not desired
            phi = np.array(phi)			
        if(kernel=="BF"):
            phi = list()
            for i in range(len(self.x_set)):
                for value in self.x_set[i]:
                    if(fix_index==False or i in index_set):
                        phi.append(1.0*(x[i]==value))
            #phi.append(1) # disable this if a constant is not desired
            phi = np.array(phi)
        if(kernel=="quadBF"):
            phi = list()
            '''
            for i in range(len(self.x_set)):
                for value in self.x_set[i]:
                    if(fix_index==False or i in index_set):
                        phi.append(1.0*(x[i]==value))
            '''
            for i in range(len(self.x_set)):
                for value in self.x_set[i]:
                    for j in range(i+1,len(self.x_set)):
                        for value_j in range(len(self.x_set[j])):
                            if(fix_index==False or (
                                    i in index_set and j in index_set)):
                                phi.append(1.0*
                                       (x[i]==value)*(x[j]==value_j))
            
            phi = np.array(phi)
            
        return phi
        
    def _ratiogroup(self,x,kernel="Lin"):
        fix_index  = self.fix_index
        index_set = self.index_set
        if(kernel=="Lin"):
            groupx = list()
            for i in range(len(x)):
                t = [i]
                groupx.append(t)
        if(kernel=="quadsingle"):
            groupx = list()		
            for i in range(len(x)):
                t = [i,i+len(x)]
                groupx.append(t)
        if(kernel=="Binary"):
            groupx = list()
            for i in range(len(x)):
                z1 = list()
                z1.append(2*i)
                z1.append(2*i+1)	
                groupx.append(z1)	
        if(kernel=="BF"):
            index = 0
            groupx = list()
            for i in range(len(self.x_set)):
                z1 = list()
                for value in self.x_set[i]:
                    if(fix_index==False or i in index_set):
                        z1.append(index)
                        index+=1
                groupx.append(z1)	
        if(kernel=="BFFix"):
            index = 0
            groupx = list()
            for i in range(len(self.x_set)):
                z1 = list()
                for value in self.x_set[i]:
                    if(fix_index==False or i in index_set):
                        z1.append(index)
                        index+=1
                groupx.append(z1)					
            phi = list()	
            j = 0			
            for i in range(len(self.x_set)):
                if(not(i in index_set)):
                    continue					
                for value in self.x_set[i]:
                    if(1):
                        #print("x[j]",j,value,index_set,x)					
                        phi.append(1.0*(x[j]==value))
                j += 1 						
            #phi.append(1) # disable this if a constant is not desired
            phi = np.array(phi)	
			
        if(kernel=="quadBF"):
            index = 0
            groupx = list()
            for i in range(len(self.x_set)):
                z1 = list()
                groupx.append(z1)
                
            '''
            for i in range(len(self.x_set)):
                for value in self.x_set[i]:
                    if(fix_index==False or i in index_set):
                        groupx[i].append(index)
                        index+=1
            '''
            for i in range(len(self.x_set)):
                for value in self.x_set[i]:
                    for j in range(i+1,len(self.x_set)):
                        for value_j in range(len(self.x_set[j])):
                            if(fix_index==False or (
                                    i in index_set and j in index_set)):
                                groupx[i].append(index)
                                groupx[j].append(index)
                                index+=1
        #print("group index number:", index)					
        return groupx 
    
 
class Seesd(ShiftEstimator):
    def __init__(self,
	             s=1,
				 soft=False,):
        self.s = s	  
        self.soft = soft
        self.indexes_learned = []
        self.max_iter=20
        return 
    
    def _estimate_shift(self,
					   source=None,
                       target=None,
                       source_yhat=None,
                       target_yhat=None,
                       source_proba = None,
                       target_proba = None,
                       source_y = None,
					   verbose=False,
					   ):
        '''
        shift estimation method
        parameter:
            source: n by d array, source feature data
            target: m by d array, target feature data
            source_yhat: n by 1 array, predicted labels on source data 
            target_yhat: m by 1 array, predicted labels on target data
            source_proba: n by L array, predicted label prob on source data
            target_proba: m by L array, predicted label prob on source data
            
        return: 
            shiftfeature: a list of indices, indicating the shifted features
            shift: estimated performance differences between target and source.
        It equals to acc(target) - acc(source).
        '''
        
#        verbose=True
        max_iter = self.max_iter
#        max_iter = 20		
        s = self.s			
        soft = self.soft
		
        self.verbose = verbose
        eps_min = 1e10
        error_tol = eps_min
        # Initialize the Indexset
        d = target.shape[1]
        I = set(range(d))        
        s_subset = [set(i) for i in itertools.combinations(range(d), s)]
        
        
        #print("unique values", len(np.unique(target[:,7])))
        target = np.asarray(target)
        s1 = np.asarray(source)
        #print("target",set((target[:,7].flatten())))
        total = np.concatenate((target,s1))		
        #print("total",total.shape,np.unique(total,0),total)
        X_sizes1 = [max(np.unique(total[:,i])) for i in range(d)]
        X_sizes2 = [min(np.unique(total[:,i])) for i in range(d)]
        X_sizes = [max(int(X_sizes1[i]-X_sizes2[i])+2,2) for i in range(d)]
        self.X_sizes = X_sizes		
        #X_sizes =  [int(1+max((set((source[:,i].flatten()).tolist()[0])).union(
        #    set((target[:,i].flatten()).tolist()[0])))) 
        #            for i in range(d)]   
#        i=7
        #print("line 243 X sizes", X_sizes,X_sizes1,X_sizes2)
        #print("shapes of source", source.shape)
        s1 = np.asarray(source)
        #print("unique values for y ", len(np.unique(s1[:,-1])))
        #print("unique values for yhat", len(np.unique(source_yhat)))

        
        yhat_size = len(np.unique(source_yhat))
        y_size =  len(np.unique(np.asarray(source_y)))

        yhat_size = y_size		
        self.y_size = y_size	
        self.yhat_size=yhat_size		
        ExistFeatureShift = False
        i=0
       # Prune the indexset
        for subset in s_subset:
            i+=1
            if(i<-1):
                break
#            subset={9,2,4}
#            subset={1,2,7}			
            if(self.verbose==True):
                print("line 251 subset",subset)
            
            if(I.issubset(subset)):
                continue
            if(ExistFeatureShift and bool(I&subset)==False ):
                continue
            #subset={6}
            # Estimate the probability
            selected_index = list(sorted(subset))
            selected_size = [X_sizes[i] for i in selected_index]
            #print("line 243 selected indexes",selected_index, X_sizes,selected_size,y_size,yhat_size)
            ps_x_y, ps_x_yhat_y, pt_x_yhat,ps_x_yhat = self._InitializeProb(
                selected_size, y_size, yhat_size)
            #print("line 268", ps_x_y.shape)
            self._EstimateDist(source=source, 
								source_y=source_y,
                               target=target,
                               source_yhat=source_yhat,
                               target_yhat=target_yhat,
							   target_y_proba = target_proba,
							   source_y_proba=source_proba,
                               selected_index=selected_index,
                               ps_x_y=ps_x_y, 
                               ps_x_yhat_y=ps_x_yhat_y, 
                               pt_x_yhat=pt_x_yhat,
							   ps_x_yhat=ps_x_yhat,
							   soft=soft,
							   )
            
            # Get the weight W and the error

            #print("size of solve weight system")
            #print("ps_x_y",ps_x_y.shape,ps_x_yhat_y.shape,pt_x_yhat.shape)			
            W, Error = self.SolveWeightSystem(ps_x_y,
                          ps_x_yhat_y,
                          pt_x_yhat,
						  ps_x_yhat)
            
            Error_full = self.EstimateError(source=source,
											source_y=source_y,
                                            target=target,
                                            source_yhat=source_yhat,
                                            target_yhat=target_yhat,
											target_y_proba=target_proba,
											source_y_proba=source_proba,
                                            W=W, 
                                            s_subset=s_subset,
                                            this_set=subset,
                                            tol=error_tol,
        				                    max_iter=max_iter,
											soft=soft)
#            Error_full += Error											
            if(verbose==True):               			
                print("line 273 subset and error", subset, Error_full,Error) # Debug only

            if(Error_full<=eps_min or i<=1):
                eps_min = Error_full
                U_best = subset
                W_best = W
                error_tol = Error_full
            if(Error_full>=error_tol):
                ExistFeatureShift = True
            #print("line 277 Full Error", subset, I,(I&subset), Error_full, Error)
            if(Error_full<error_tol):
                I = I.intersection(subset)  # Prune the indeset set      
        
        # Estimate the error for the given W
        I = U_best
        W = W_best
               
        # Estimate the importance weight.
        self.Est_Shift = I,W,eps_min
        self.indexes_learned = list(I)		
        if(verbose==True):
            print("result of Sees-d",I,W,eps_min)
        return I, W, eps_min
		
    def getindex(self,s):		
        indexes_learned =self.indexes_learned
        while(len(indexes_learned)<s):
            indexes_learned.append(-1)
        return indexes_learned	
		
    def EstimateError(self,
                      source,
					  source_y,
                      target,
                      source_yhat,
                      target_yhat,
                      W,
                      this_set,
                      s_subset,
                      tol,
                      use_random=True,
                      max_iter=20,
					  target_y_proba=None,
					  source_y_proba=None,
					  soft=False):
        d = target.shape[1]
        
        #X_sizes =  [int(1+max((set((source[:,i].flatten()).tolist()[0])).union(
        #    set((target[:,i].flatten()).tolist()[0])))) 
        #            for i in range(d)]   
        
        X_sizes = self.X_sizes
#        X_sizes = [len(np.unique(target[:,i])) for i in range(d)]

        yhat_size = len(set((source_yhat.flatten()).tolist()))
        y_size =  len(set((source_y[:,-1].flatten()).tolist()[0]))
         
        y_size = self.y_size
        yhat_size = self.yhat_size		 
		
        #print("y_size is",y_size)				  
        error = 0
        subset_index_random = random.sample(range(len(s_subset)),len(s_subset))
        for i in range(len(s_subset)):
            if(use_random==False):
                cs = s_subset[i]
            else:
                if(i>=max_iter):
                    break
                cs = s_subset[subset_index_random[i]]
               
            #cs={6}
            subset = this_set.union(cs)
            # Estimate the probability
            selected_index = list(sorted(subset))
            selected_size = [X_sizes[i] for i in selected_index]
            #print("line 312", X_sizes,selected_size,y_size,yhat_size)
            ps_x_y, ps_x_yhat_y, pt_x_yhat, ps_x_yhat = self._InitializeProb(
                selected_size, y_size, yhat_size)
            self._EstimateDist(source=source, 
								source_y=source_y,
                               target=target,
                               source_yhat=source_yhat,
                               target_yhat=target_yhat,
							   target_y_proba =  target_y_proba,
							   source_y_proba=source_y_proba,
                               selected_index=selected_index,
                               ps_x_y=ps_x_y, 
                               ps_x_yhat_y=ps_x_yhat_y, 
                               pt_x_yhat=pt_x_yhat,
							   ps_x_yhat=ps_x_yhat,
							   soft=soft,
							   )
            
            # Get the weight W and the error
            #time1 = time.time()
            error_onetime = self.GetErrorFixSubset(W=W,
                                                   ps_x_y=ps_x_y,
                          ps_x_yhat_y=ps_x_yhat_y,
                          pt_x_yhat=pt_x_yhat,
                          selected_index=selected_index,
                          W_set=this_set)
            #time2 = time.time()-time1
            #print("linre 373 fix subset runtime", time2)
            #print("line 346", error_onetime,cs,s_subset)
            error+=error_onetime/len(s_subset)
            #print("line 357", i)
            #if(error_onetime>tol):
            #    print("early stop error one time",i)
            #    return error_onetime
            if(error>tol):
                #print("early stop error larget")
                return error
            
            
        return error

    def _EstimateDist(self,
                      source, 
                      target,
                      source_yhat,
                      target_yhat,
                      selected_index,
                      ps_x_y, ps_x_yhat_y, pt_x_yhat, ps_x_yhat,
					  source_y,
					  soft=False,
					  target_y_proba=None,
					  source_y_proba=None,):
        for i in range(len(source)):
            x_index = [int(source[i,j]) for j in selected_index]
            index = copy.copy(x_index)
            index.append(int(source_y[i]))
            index = tuple(index)
            #print("line 309",index,ps_x_y,"select", ps_x_y[index])
            #ps_x_y.itemset((2,0),1)
            ps_x_y[index] += 1/len(source)     
            #print("line 312",index,ps_x_y,"select", ps_x_y[index])
            if(soft==False):            			
                x_index.append(int(source_yhat[i]))
                ps_x_yhat[tuple(x_index)] += 1/len(source)
                x_index.append(int(source_y[i]))
                index  = tuple(x_index)
                #print("line 320",index,ps_x_yhat_y,"select")
                ps_x_yhat_y[index] += 1/len(source)    
            else:
                ps_x_yhat[tuple(x_index)] += source_y_proba[i]/len(source)
                for temp in range(len(source_y_proba[i])):
                    index = copy.copy(x_index)
                    index.append(temp)
                    index.append(int(source_y[i]))
                    ps_x_yhat_y[tuple(index)] += source_y_proba[i,temp]/len(source)      
        for i in range(len(target)):
            x_index = [int(target[i,j]) for j in selected_index]
            if(soft==False):
                x_index.append(int(target_yhat[i]))
                index = tuple(x_index)
                pt_x_yhat[index] += 1/len(target)
            else:
                index = tuple(x_index)
                pt_x_yhat[index] += target_y_proba[i]/len(target)   
        return 0

        
    def SolveWeightSystem(self,
                          ps_x_y,
                          ps_x_yhat_y,
                          pt_x_yhat,
						  ps_x_yhat):

        eps = 1e-10
        
        x_size = ps_x_y.shape
        yhatsize = ps_x_yhat.shape[-1]
        #print("line 336",x_size,ps_x_y.shape)
        x_size1 = [x_size[i] for i in range(len(x_size)-1)]
        
        # Form objective
        obj = 0
        #print("line 429",ps_x_y.shape[0:-1])
        W_full = np.empty(tuple(ps_x_y.shape[0:-1]), dtype=cp.Variable)
#        W = cp.Variable(ps_x_y.shape)
        
        x_all = GetAllPerm(value_set=x_size1,start=[])
        for x_index in x_all:
            
            pt1 = pt_x_yhat[tuple(x_index)]
            ps1 = ps_x_yhat_y[tuple(x_index)]
            p_cond = ps_x_yhat[tuple(x_index)]
            has0 = sum(p_cond==0)
            scale1 = np.asmatrix(p_cond).transpose()*np.ones((1,yhatsize))		
            ps1_scale = ps1/(scale1+has0*eps)
            pt1_scale = pt1/(p_cond+has0*eps)			
            if(W_full[tuple(x_index)]==None):
                W_full[tuple(x_index)] = cp.Variable(ps_x_y.shape[-1])
            W1 = W_full[tuple(x_index)]
            #pt1-cp.ps1           
            if(self.verbose==True):
                print("line 345 is", has0, ps1,p_cond, ps1_scale, np.linalg.cond(ps1_scale),np.linalg.cond(ps1))
            obj += cp.sum_squares(ps1_scale @ W1 - pt1_scale)         
#            obj += cp.sum_squares(ps1 @ W1 - pt1)         
            
        # Constraints  
        constraints = [
#            W>=0,
#            cp.sum( cp.multiply(W, ps_x_y))==1,
            ]
        sum1 = 0
        for x_index in x_all:
            constraints+=[W_full[tuple(x_index)]>=0]
            ps1 = ps_x_y[tuple(x_index)]
            sum1+= cp.sum(cp.multiply(W_full[tuple(x_index)], ps1))
            
        constraints+=[sum1==1
                      ]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        try:		
            problem.solve(verbose=self.verbose,solver=cp.MOSEK)        
        except:
            if(self.verbose==True):
                print("MOSEK did not work. try default solver")
            problem.solve(verbose=self.verbose)        			
        W_values = np.zeros(ps_x_y.shape)
        for x_index in x_all:
            W_values[tuple(x_index)] = W_full[tuple(x_index)].value
        
        #print("line 525",np.nan_to_num(W_values,nan=1),W_values,ps_x_y,ps_x_yhat_y,pt_x_yhat,ps_x_yhat)	
        W_values = np.nan_to_num(W_values,nan=1)		
        return W_values, problem.value 

    def _InitializeProb(self,
                        x_size, y_size, yhat_size):
        s_size = copy.copy(x_size)
        s_size.append(y_size)
        ps_x_y = np.zeros(s_size) 
        pt_x_yhat = np.zeros(s_size)
        ps_x_yhat = np.zeros(s_size)
        s_size = copy.copy(x_size)
        s_size.append(yhat_size)
        s_size.append(y_size)
        ps_x_yhat_y = np.zeros(s_size)
        return ps_x_y, ps_x_yhat_y, pt_x_yhat, ps_x_yhat
		
    def GetErrorFixSubset(self,
                          W,
                          ps_x_y,
                          ps_x_yhat_y,
                          pt_x_yhat,
                          selected_index,
                          W_set):

        x_size = ps_x_y.shape
        #print("line 351", ps_x_y.shape, selected_index)
        x_size1 = [x_size[i] for i in range(len(x_size)-1)]
        
        # Compute the square error
        error = 0 
        
        x_all = GetAllPerm(value_set=x_size1,start=[])
        
        W_indexmap = [j for j in range(len(selected_index)) 
                      if selected_index[j] in W_set]
 
            
        # List comprehension speed-ups
        pt1 = [ pt_x_yhat[tuple(x_index)] for x_index in x_all ]
        ps1 = [ ps_x_yhat_y[tuple(x_index)] for x_index in x_all ]
        W_index = [ [x_index[j] for j in W_indexmap] for x_index in x_all ]
        error = [ np.sum((ps1[j]@W[tuple(W_index[j])]-pt1[j])**2) for j in range(len(pt1))]
        error = np.sum(error)	
        return error
		
    def predict_weights(self, \
                        x_samples=None,
                        y_samples=None,):
        source = x_samples
        #y = y_samples
        Est_Shift = self.Est_Shift
	
        x_indexes = tuple(list(sorted(Est_Shift[0])))
        
        weights = list()
        for i in range(len(source)):
            y = y_samples[i]
            #y = source[i,-1]
            #yhat = source_yhat[i]
                        
            if(1):
                value_x = (np.asarray(source)[i,x_indexes])
                value_x =  value_x.astype(int)
                value_y = int(y)
                indexes = value_x.tolist()
                indexes.append(value_y)
                indexes  = tuple(indexes)
                ratio = float(Est_Shift[1][indexes])   
            weights.append(ratio)
        return weights

def GetAllPerm(value_set:list(), start=[]):
    if(len(start)==len(value_set)):
        #print("start is", [start])
        return [copy.copy(start)]
    # break condition
    result_all = list()    
    # go over all posibility for this position
    ptr = len(start)
    for i in range(value_set[ptr]):
        start0 = copy.copy(start)
        start0.append(i)
        result = GetAllPerm(value_set, start0)
        result_all += result
    return result_all
    
def main():
    dir1 = "temp/"	
    source, target, source_yhat, target_yhat, source_proba,  \
    target_proba = util.load_data(
        source_path=dir1+"source.csv",
             target_path=dir1+"target.csv",
			 source_yhat_path=dir1+"source_yhat.csv",
			 target_yhat_path=dir1+"target_yhat.csv",
			 source_proba_path=dir1+"source_proba.csv",
			 target_proba_path=dir1+"target_proba.csv",)
    #'''
       
    print("source",source[0])
    print("finish")
    SSEst = Seesc()

    SSEst.set_params(#verbose=True,
                         eta = 0.005, # 0.22 , 0.3 with quadsingle kernel is good for CA->PR
                         #kernel="quadBF",
                         kernel="BF",
						 #kernel="Lin",
						 #kernel="quadsingle",
                         fix_index=False,
                         #index_set=[5],
						 post_sparse=False,
                         s=1)	
    
    SSEst = Seesd()


    result = SSEst.estimate_shift(source=source[:,0:-1],
                         target=target[:,0:-1],
                         source_y=source[:,-1],
                         source_yhat=source_yhat,
                         target_yhat=target_yhat,
                         source_proba = source_proba,
                         target_proba = target_proba,
                         )
    print(result)
    print("result is", result)
    weights = SSEst.predict_weights(source[:,0:-1],source[:,-1])
    acc_source = [source[i,-1]==source_yhat[i] for i in range(len(source))]
    acc = eval_performance(weights,acc_source)
    print("est acc is", acc)

    print("true source acc is", compute_acc(source[:,-1],source_yhat) )
    print("true target acc is", compute_acc(target[:,-1],target_yhat) )
    return
 

if __name__ == '__main__':
    main()           