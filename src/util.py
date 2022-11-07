import numpy as np
from sklearn.metrics import mean_squared_error
import numpy

def eval_performance(weights=None, 
                     acc=None):
    acc1 = [weights[i]*acc[i] for i in range(len(acc))]
    return sum(acc1)/len(acc1)
    
def eval_kl_dist(weights_true=None,
                 weights_est=None):
    #print("weights_true",weights_true)
    eps=1e-10	
    dist = [np.log(weights_true[i]/(eps+weights_est[i])) for i in range(len(weights_true))]
    return sum(dist)/len(dist)

def eval_mse_dist(weights_true=None,
                 weights_est=None):
#    print("est weights is",weights_est,type(weights_est))				 
    weights_est = np.asarray(weights_est)	
    weights_est[np.isnan(weights_est)] = 1				 
#    print("est weights is",weights_est)
    return mean_squared_error(weights_true,weights_est)
	
def weight_retrain(x_samples=None,
                   y_samples=None,
                    weights=None,
                   model=None):
    model.fit(x_samples,y_samples, sample_weight = weights)
    return model

def compute_acc(labels=None,pred_labels=None):
    acc = [labels[i]==pred_labels[i] for i in range(len(labels))]
    return sum(acc)/len(acc)

def compute_true_weights_old(x_samples,y_samples,p_t, p_s,index_set):
    weights = list()
    for i in range(len(x_samples)):
        full_values = np.asarray(x_samples[i]).flatten()
        sel_values = list()
        for j in range(len(index_set)):
#            print("full value is",full_values)
            k = full_values[index_set[j]]		
            sel_values.append(int(k))		
        sel_values.append(int(y_samples[i]))		
        values = tuple(sel_values)
#        print("values are", values)
        a = p_t[values]/p_s[values]
        weights.append(a)		
    return weights

def compute_true_weights(x_samples,y_samples,params):
    weights = list()
    source_dict = dict()	
    for i in range(len(x_samples)):
        full_values = np.asarray(x_samples[i]).flatten()
        sel_values = list()
        column_maps = params["column_maps"]
        feature_maps = params["value_maps"]
        shifts = params["shift"]
        shift_type = params["type"]
        if( shift_type=="sep"):
            value1 = 0
            for a in shifts:
                index1 = column_maps[a] # shift index 
                if(index1<len(full_values)):
                    x1 = full_values[index1] # shift values
                else:
                    x1 = y_samples[i]
                for j in shifts[a]: 			
                    if(x1==feature_maps[j]): # map to the right value
                        break
                x1 = float(x1)	 					
#                print("index and x1",index1,float(x1))        						
                if(index1<len(full_values)):
                    if(not (index1,x1) in source_dict):
                        source_dict[(index1,x1)] = float(sum(x_samples[:,index1]==x1)/len(x_samples))
                    ratio_s = source_dict[(index1,x1)]
                else:
                    if(not (index1,x1) in source_dict):
                        source_dict[(index1,x1)] = float(sum(y_samples==x1)/len(y_samples))					
                    ratio_s = source_dict[(index1,x1)]
        
                value1 += shifts[a][j]/ratio_s
            weights.append(value1)
        elif( shift_type=="BF"):
            source_dist = params["source_dist"]
            target_dist = params["target_dist"]

            x_full = list(full_values)
            x_full.append(y_samples[i])
            shift_dims = params["shift_dims"]
            shift_keys = list(shift_dims.keys())	
            feature_maps = params["maps"]
            column_maps	= 	params["column_maps"]	
            #print("x full size",len(x_full))
			
            s_weight = get_dist_BF(x_full,
     			dist = params["source_dist"],
				shift_dims=shift_dims,
				shift_keys=shift_keys,
				feature_maps=feature_maps,
				column_maps=column_maps)

            t_weight = get_dist_BF(x_full,
     			dist = params["target_dist"],
				shift_dims=shift_dims,
				shift_keys=shift_keys,
				feature_maps=feature_maps,
				column_maps=column_maps)
				
            value1 = t_weight/s_weight
            weights.append(value1)
        elif( shift_type=="2BF"):
            source_dist = params["source_dist"]
            target_dist = params["target_dist"]

            x_full = list(full_values)
            x_full.append(y_samples[i])
            shift_dims = params["shift_dims"]
            shift_keys = list(shift_dims.keys())	
            feature_maps = params["maps"]
            column_maps	= 	params["column_maps"]	
            #print("x full size",len(x_full))
			
            s_weight = get_dist_2BF(x_full,
     			dist = params["source_dist"],
				shift_dims=shift_dims,
				shift_keys=shift_keys,
				feature_maps=feature_maps,
				column_maps=column_maps)

            t_weight = get_dist_2BF(x_full,
     			dist = params["target_dist"],
				shift_dims=shift_dims,
				shift_keys=shift_keys,
				feature_maps=feature_maps,
				column_maps=column_maps)
				
            value1 = t_weight/s_weight
            weights.append(value1)	
        elif( shift_type=="3BF"):
            source_dist = params["source_dist"]
            target_dist = params["target_dist"]

            x_full = list(full_values)
            x_full.append(y_samples[i])
            shift_dims = params["shift_dims"]
            shift_keys = list(shift_dims.keys())	
            feature_maps = params["maps"]
            column_maps	= 	params["column_maps"]	
            #print("x full size",len(x_full))
			
            s_weight = get_dist_3BF(x_full,
     			dist = params["source_dist"],
				shift_dims=shift_dims,
				shift_keys=shift_keys,
				feature_maps=feature_maps,
				column_maps=column_maps)

            t_weight = get_dist_3BF(x_full,
     			dist = params["target_dist"],
				shift_dims=shift_dims,
				shift_keys=shift_keys,
				feature_maps=feature_maps,
				column_maps=column_maps)
				
            value1 = t_weight/s_weight
            weights.append(value1)				
        elif(shift_type=="Mix"):
            weights.append(1) # TODO: fix this       
        else:
            raise NotImplementedError	
        #print("i is",i)
    return weights

def get_dist_BF(x_full,
     			dist,
				shift_dims,
				shift_keys,
				feature_maps,
				column_maps,
				):
        full_part = list()
        index1 = column_maps[shift_keys[0]]
        index2 = column_maps[shift_keys[1]]			
        x1 = x_full[index1]
        x2 = x_full[index2]		
        for i1 in dist: # first dimension
            s1 = dist[i1]
            f1 = shift_keys[0]		
            for j1 in s1: # second dimension
                s2 = s1[j1]
                f2 = shift_keys[1]  
                #print("feature map",feature_maps,f1,f2)		
                if(feature_maps[i1]== x1 and feature_maps[j1] == x2):
                    return s2
        print("data loader error, does not find the value")					
        return None

def get_dist_2BF(x_full,
     			dist,
				shift_dims,
				shift_keys,
				feature_maps,
				column_maps,
				):
        full_part = list()
        index1 = column_maps[shift_keys[0]]
        index2 = column_maps[shift_keys[1]]			
        index3 = column_maps[shift_keys[2]]
        x1 = x_full[index1]
        x2 = x_full[index2]		
        x3 = x_full[index3]		
        for i1 in dist: # first dimension
            s1 = dist[i1]
            f1 = shift_keys[0]		
            for j1 in s1: # second dimension
                s2 = s1[j1]
                f2 = shift_keys[1] 
                for k1 in s2:
                    s3 = s2[k1]				
                    #print("feature map",feature_maps,f1,f2)		
                    if(feature_maps[i1]== x1 and feature_maps[j1] == x2 and feature_maps[k1] == x3):
                        return s3
        print("data loader error, does not find the value")					
        return None

def get_dist_3BF(x_full,
     			dist,
				shift_dims,
				shift_keys,
				feature_maps,
				column_maps,
				):
        full_part = list()
        index1 = column_maps[shift_keys[0]]
        index2 = column_maps[shift_keys[1]]			
        index3 = column_maps[shift_keys[2]]
        index4 = column_maps[shift_keys[3]]
        x1 = x_full[index1]
        x2 = x_full[index2]		
        x3 = x_full[index3]		
        x4 = x_full[index4]		
        for i1 in dist: # first dimension
            s1 = dist[i1]
            f1 = shift_keys[0]		
            for j1 in s1: # second dimension
                s2 = s1[j1]
                f2 = shift_keys[1] 
                for k1 in s2:
                    s3 = s2[k1]	
                    for l1 in s3:
                        s4 = s3[l1]						
                    #print("feature map",feature_maps,f1,f2)		
                        if(feature_maps[i1]== x1 and feature_maps[j1] == x2 and feature_maps[k1] == x3 and feature_maps[l1]==x4):
                            return s4
        print("data loader error, does not find the value")					
        return None
		
def load_data(source_path="../data/adult/adult_s.csv",
             target_path="../data/adult/adult_t.csv",
			 source_yhat_path="",
			 target_yhat_path="",
			 source_proba_path="",
			 target_proba_path="",
			 ):
    '''
    Directly load data from files.
    '''
    source = numpy.loadtxt(source_path,delimiter=",")
    target = numpy.loadtxt(target_path,delimiter=",")
    source_yhat = numpy.loadtxt(source_yhat_path,delimiter=",")
    target_yhat = numpy.loadtxt(target_yhat_path,delimiter=",")
    source_proba = numpy.loadtxt(source_proba_path,delimiter=",")
    target_proba = numpy.loadtxt(target_proba_path,delimiter=",")
		
    source = numpy.asmatrix(source)		
    target = numpy.asmatrix(target)
    return source, target, source_yhat, target_yhat, source_proba, target_proba
    
def main():
    weights = [1,2,1]
    acc = [0,0,1]
    print("eval result",eval_performance(weights,acc))
    return 
 

if __name__ == '__main__':
    main()           