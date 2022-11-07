from sklearn.ensemble import GradientBoostingClassifier
from dataloader import DataLoader
from util import compute_acc
import json

from sees import Seesc, Seesd
from baselines import Bbse, Klieps,DLCS

def shift_estimation_example(
        datasize_source=5000,
        datasize_target=5000,
        paramspath = "../shiftparam/BankchurnBfGeo999.json",
        datapath="../dataset/bankchurn/bankchurn.csv",
        dataset="covid",
        random_seed=0,
        s=1):
    '''
    shift estimation example.
    parameters:
        datasize_source: scalar, size of the source data
        datasize_target: scalar, size of the target data
        paramspath: str, path to the shift param to generate the data
        datapath: str, dataset path
        dataset: str, name of the dataset
        random_seed: int, random seed
        s: int, sparsity parameter
    return:
    '''
    print("example on dataset: ", dataset)
    # 1. specify ML model
    model = GradientBoostingClassifier(n_estimators=50)	
        
    # 2. create dataset
    f = open(paramspath)
    shift_params = json.load(f)	
    params = shift_params			
    params["model"] = model
    params["source_size"] = datasize_source # args.sourcesize	
    params["target_size"] = datasize_target # args.targetsize
    params["random_state"] = random_seed
    params["shift_dist"] = shift_params["shift"]
    params["maps"] = shift_params["value_maps"]
    params["datapath"] = datapath
    MyLoader = DataLoader()
    source, target, source_yhat, target_yhat, \
                source_proba, target_proba \
                = MyLoader.generate(params=params,
								  data_name=dataset)   
    # 3. estimate and explain performance
    
    SSEst = Bbse()  # switch to Bbse
    SSEst = DLCS()  # switch to DLCS
    SSEst = Klieps()  # switch to Klieps
    SSEst = Seesc() # switch to SEES-c
    SSEst = Seesd() # switch to SEES-d




    SSEst.set_params(#verbose=True,
                     eta = 0.01,                      #kernel="quadBF",
                         kernel="BF",
						 #kernel="Lin",
						 fix_index=False,
                         post_sparse=False,
                         s=s)
       
    result_est = SSEst.estimate_shift(source=source[:,0:-1],
                         target=target[:,0:-1],
                         source_y=source[:,-1],
                         source_yhat=source_yhat,
                         target_yhat=target_yhat,
                         source_proba = source_proba,
                         target_proba = target_proba,
                         )
    print("estimated shifted features:", result_est[0])
    print("estimated performance shift:", result_est[1])
    
    # 4. evaluate true performance
    acc_s = compute_acc(source[:,-1],source_yhat) 
    acc_t = compute_acc(target[:,-1],target_yhat)
    gap = acc_t[0]-acc_s[0]
    print("true performance shift:", gap)
    return (result_est[1]-gap)*(result_est[1]-gap)

def main():
    '''
    shift_estimation_example(
        datasize_source=5000,
        datasize_target=5000,
        paramspath = "../shiftparam/BankchurnBfGeo999.json",
        datapath="../dataset/bankchurn/bankchurn.csv",
        dataset="bankchurn",
        random_seed=0,
        s=1,)
    '''
    
    '''
    shift_estimation_example(
        datasize_source=5000,
        datasize_target=5000,
        paramspath = "../shiftparam/CreditBfMar999.json",
        datapath="../dataset/credit/credit.csv",
        dataset="credit",
        random_seed=0,
        s=1,)
    '''
    
    shift_estimation_example(
        datasize_source=5000,
        datasize_target=5000,
        paramspath = "../shiftparam/CovidBFAge999.json",
        datapath="../dataset/covid/corona_202201.csv",
        dataset="covid",
        random_seed=0,
        s=1)
        
if __name__ == '__main__':
    main()  