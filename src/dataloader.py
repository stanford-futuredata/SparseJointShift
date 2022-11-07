# data loader for kiss
from sklearn.model_selection import train_test_split
import numpy
from pandas import read_csv
import pandas

class DataLoader(object):
    def __init__(self):
        self.pums_names = ["employ","income","public","mobile","traveltime"]	
        #self.pums_namestime = ["timeemploy","timeincome","timepublic","timemobile","timetraveltime"]	
		
        return 

    def generate(self,
	             data_name="employ",
				 params=None):
        self.data_name = data_name	
        #print("data name is: ",data_name)		
						
        if(params["type"]=="2BF"):
            return self._generate_2BF(params=params)				
        if(params["type"]=="3BF"):
            return self._generate_3BF(params=params)			
        if(params["type"]=="BF"):
            return self._generate_BF(params=params)		
			
        return 				 

    def _generate_from_dist(self,
							dataframe,
							dist,
							shift_dims,
							shift_keys,
							full_size,
							random_state,
							):
        full_part = list()
        #print("dist is ",dist)
        for i1 in dist:
            s1 = dist[i1]
            f1 = shift_keys[0]		
            for j1 in s1:
                s2 = s1[j1]
                f2 = shift_keys[1]   
                #print("dataframe is:",dataframe)
                M0 = dataframe.loc[(dataframe[f1] == i1)] 	             	
                M0 = M0.loc[M0[f2]==j1]				
                #print("s is",f1,f2,i1,j1,s2)
                sizeM1 = int(2*full_size*s2) # doubled for training	
                M1 = M0.sample(n = sizeM1,random_state=random_state,replace=True)
                full_part.append(M1)
        return full_part			

    def _generate_from_dist2(self,
							dataframe,
							dist,
							shift_dims,
							shift_keys,
							full_size,
							random_state,
							s=2):
        full_part = list()
        #print("dist is for BF2",dist)
        for i1 in dist:
            s1 = dist[i1]
            f1 = shift_keys[0]		
            for j1 in s1:
                s2 = s1[j1]
                f2 = shift_keys[1]   
                #print("dataframe is:",dataframe)
                M0 = dataframe.loc[(dataframe[f1] == i1)] 	             	
                M0 = M0.loc[M0[f2]==j1]	
                for k1 in s2:
                    s3 = s2[k1]
                    f3 = shift_keys[2]
                    M1 = M0.loc[M0[f3]==k1]					
                    #print("s is",f1,f2,f3,i1,j1,k1,s3)
                    sizeM1 = int(2*full_size*s3) # doubled for training	
                    M_full = M1.sample(n = sizeM1,random_state=random_state,replace=True)
                    full_part.append(M_full)
        return full_part

    def _generate_from_dist3(self,
							dataframe,
							dist,
							shift_dims,
							shift_keys,
							full_size,
							random_state,
							s=2):
        full_part = list()
        #print("dist is for BF2",dist)
        for i1 in dist:
            s1 = dist[i1]
            f1 = shift_keys[0]		
            for j1 in s1:
                s2 = s1[j1]
                f2 = shift_keys[1]   
                #print("dataframe is:",dataframe)
                M0 = dataframe.loc[(dataframe[f1] == i1)] 	             	
                M0 = M0.loc[M0[f2]==j1]	
                for k1 in s2:
                    s3 = s2[k1]
                    f3 = shift_keys[2]
                    M1 = M0.loc[M0[f3]==k1]
                    for l1 in s3:
                        s4 = s3[l1]
                        f4 = shift_keys[3]
                        M2 = M1.loc[M1[f4]==l1]							
                        #print("s is",f1,f2,f3,f4,i1,j1,k1,l1,s4)
                        sizeM1 = int(2*full_size*s4) # doubled for training	
                        M_full = M2.sample(n = sizeM1,random_state=random_state,replace=True)
                        full_part.append(M_full)
        return full_part
		
    def _generate_2BF(self,params):	
        # brute force approach for generate shift
	    # load all params
	
        datapath = params["datapath"]
        #print("datapath is",datapath)
        source_size = params["source_size"]
        target_size = params["target_size"]
		
        random_state = params["random_state"]
	    # load source and target data		
        dataframe = read_csv(datapath, header=0, na_values='?')
        
        # generate source
        shift_dims = params["shift_dims"]
        shift_keys = list(shift_dims.keys())	
        feature_maps = params["maps"]		
        #print("shift dims", shift_dims.keys())
        source_dist = params["source_dist"]
        source_part = list()
		
        source_part = self._generate_from_dist2(
					dataframe=dataframe,
					dist=source_dist,
					shift_dims=shift_dims,
					shift_keys=shift_keys,
					full_size=source_size,
					random_state=random_state,
					s=2)

        target_dist = params["target_dist"]
					
        target_part = self._generate_from_dist2(
					dataframe=dataframe,
					dist=target_dist,
					shift_dims=shift_dims,
					shift_keys=shift_keys,
					full_size=target_size,
					random_state=random_state,
					s=2)
					
        source_frame = pandas.concat(source_part)					
        target_frame = pandas.concat(target_part)
					
		# transform data
        source_n = source_frame.replace(feature_maps)
        target_n = target_frame.replace(feature_maps)
		
        source_d = source_n.to_numpy()
        target_d = target_n.to_numpy()
        features_s = source_d[:,0:-1]
        labels_s = source_d[:,-1]
        features_t = target_d[:,0:-1]
        labels_t = target_d[:,-1]
        #print("loaded source and target size",
		#len(features_s),len(features_t))

		# train model
        model = params["model"]
        source_size = params["source_size"]
        target_size = params["target_size"]
        X_train, X_test, y_train, y_test = train_test_split(
        features_s, labels_s, 
		test_size=min(source_size,int(len(labels_s)-1)), 
		random_state=random_state)

        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        features_t, labels_t, 
		test_size=min(target_size,int(len(labels_t)-1)), 
		random_state=random_state)
	    
        model.fit(X_train, y_train)
        score_s_te = model.score(X_test,y_test)
        score_s_tr = model.score(X_train,y_train)		
        score_t = model.score(X_test_t,y_test_t)	
        #print("size of the source and target test",len(X_test),len(X_test_t))		
        #print("acc on source train, source test, and target",
		#      score_s_tr,score_s_te,score_t)
        #print("y label for source and target",sum(y_test)/len(y_test), sum(y_test_t)/len(y_test_t))	  
        # generate the required source target inputs
        source = numpy.concatenate(((X_test),
									numpy.matrix(y_test).transpose())
									,1)    
        target = numpy.concatenate(((X_test_t),
									numpy.matrix(y_test_t).transpose()),1)
        source_yhat =  model.predict(X_test)
        target_yhat =  model.predict(X_test_t)
        source_proba = model.predict_proba(X_test)
        target_proba = model.predict_proba(X_test_t)

        SaveData(source=source,
		target=target,
        source_yhat=source_yhat,
		 target_yhat=target_yhat,
		source_proba=source_proba,
		target_proba=target_proba,
   	 source_path="temp/source.csv",
             target_path="temp/target.csv",
			 source_yhat_path="temp/source_yhat.csv",
			 target_yhat_path="temp/target_yhat.csv",
			 source_proba_path="temp/source_proba.csv",
			 target_proba_path="temp/target_proba.csv",)
			 
        return source, target, \
		source_yhat, target_yhat, \
		source_proba, target_proba 

    def _generate_3BF(self,params):	
        # brute force approach for generate shift
	    # load all params
	
        datapath = params["datapath"]
        #print("datapath is",datapath)
        source_size = params["source_size"]
        target_size = params["target_size"]
		
        random_state = params["random_state"]
	    # load source and target data		
        dataframe = read_csv(datapath, header=0, na_values='?')
        
        # generate source
        shift_dims = params["shift_dims"]
        shift_keys = list(shift_dims.keys())	
        feature_maps = params["maps"]		
        #print("shift dims", shift_dims.keys())
        source_dist = params["source_dist"]
        source_part = list()
		
        source_part = self._generate_from_dist3(
					dataframe=dataframe,
					dist=source_dist,
					shift_dims=shift_dims,
					shift_keys=shift_keys,
					full_size=source_size,
					random_state=random_state,
					s=2)

        target_dist = params["target_dist"]
					
        target_part = self._generate_from_dist3(
					dataframe=dataframe,
					dist=target_dist,
					shift_dims=shift_dims,
					shift_keys=shift_keys,
					full_size=target_size,
					random_state=random_state,
					s=2)
					
        source_frame = pandas.concat(source_part)					
        target_frame = pandas.concat(target_part)
					
		# transform data
        source_n = source_frame.replace(feature_maps)
        target_n = target_frame.replace(feature_maps)
		
        source_d = source_n.to_numpy()
        target_d = target_n.to_numpy()
        features_s = source_d[:,0:-1]
        labels_s = source_d[:,-1]
        features_t = target_d[:,0:-1]
        labels_t = target_d[:,-1]
        #print("loaded source and target size",
		#len(features_s),len(features_t))

		# train model
        model = params["model"]
        source_size = params["source_size"]
        target_size = params["target_size"]
        X_train, X_test, y_train, y_test = train_test_split(
        features_s, labels_s, 
		test_size=min(source_size,int(len(labels_s)-1)), 
		random_state=random_state)

        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        features_t, labels_t, 
		test_size=min(target_size,int(len(labels_t)-1)), 
		random_state=random_state)
	    
        model.fit(X_train, y_train)
        score_s_te = model.score(X_test,y_test)
        score_s_tr = model.score(X_train,y_train)		
        score_t = model.score(X_test_t,y_test_t)	
        #print("size of the source and target test",len(X_test),len(X_test_t))		
        #print("acc on source train, source test, and target",
		#      score_s_tr,score_s_te,score_t)
        #print("y label for source and target",sum(y_test)/len(y_test), sum(y_test_t)/len(y_test_t))	  
        # generate the required source target inputs
        source = numpy.concatenate(((X_test),
									numpy.matrix(y_test).transpose())
									,1)    
        target = numpy.concatenate(((X_test_t),
									numpy.matrix(y_test_t).transpose()),1)
        source_yhat =  model.predict(X_test)
        target_yhat =  model.predict(X_test_t)
        source_proba = model.predict_proba(X_test)
        target_proba = model.predict_proba(X_test_t)

        SaveData(source=source,
		target=target,
        source_yhat=source_yhat,
		 target_yhat=target_yhat,
		source_proba=source_proba,
		target_proba=target_proba,
   	 source_path="temp/source.csv",
             target_path="temp/target.csv",
			 source_yhat_path="temp/source_yhat.csv",
			 target_yhat_path="temp/target_yhat.csv",
			 source_proba_path="temp/source_proba.csv",
			 target_proba_path="temp/target_proba.csv",)
			 
        return source, target, \
		source_yhat, target_yhat, \
		source_proba, target_proba 
		
    def _generate_BF(self,params):	
        # brute force approach for generate shift
	    # load all params
	
        datapath = params["datapath"]
        #print("datapath is",datapath)
        source_size = params["source_size"]
        target_size = params["target_size"]
		
        random_state = params["random_state"]
	    # load source and target data		
        dataframe = read_csv(datapath, header=0, na_values='?')
        
        # generate source
        shift_dims = params["shift_dims"]
        shift_keys = list(shift_dims.keys())	
        feature_maps = params["maps"]		
        #print("shift dims", shift_dims.keys())
        source_dist = params["source_dist"]
        source_part = list()
		
        source_part = self._generate_from_dist(
					dataframe=dataframe,
					dist=source_dist,
					shift_dims=shift_dims,
					shift_keys=shift_keys,
					full_size=source_size,
					random_state=random_state)

        target_dist = params["target_dist"]
					
        target_part = self._generate_from_dist(
					dataframe=dataframe,
					dist=target_dist,
					shift_dims=shift_dims,
					shift_keys=shift_keys,
					full_size=target_size,
					random_state=random_state)
					
        source_frame = pandas.concat(source_part)					
        target_frame = pandas.concat(target_part)
					
		# transform data
        source_n = source_frame.replace(feature_maps)
        target_n = target_frame.replace(feature_maps)
		
        source_d = source_n.to_numpy()
        target_d = target_n.to_numpy()
        features_s = source_d[:,0:-1]
        labels_s = source_d[:,-1]
        features_t = target_d[:,0:-1]
        labels_t = target_d[:,-1]
        #print("loaded source and target size",
		#len(features_s),len(features_t))

		# train model
        model = params["model"]
        source_size = params["source_size"]
        target_size = params["target_size"]
        X_train, X_test, y_train, y_test = train_test_split(
        features_s, labels_s, 
		test_size=min(source_size,int(len(labels_s)-1)), 
		random_state=random_state)

        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        features_t, labels_t, 
		test_size=min(target_size,int(len(labels_t)-1)), 
		random_state=random_state)
	    
        model.fit(X_train, y_train)
        score_s_te = model.score(X_test,y_test)
        score_s_tr = model.score(X_train,y_train)		
        score_t = model.score(X_test_t,y_test_t)	
        #print("size of the source and target test",len(X_test),len(X_test_t))		
        #print("acc on source train, source test, and target",
		#      score_s_tr,score_s_te,score_t)
        #print("y label for source and target",sum(y_test)/len(y_test), sum(y_test_t)/len(y_test_t))	  
        # generate the required source target inputs
        source = numpy.concatenate(((X_test),
									numpy.matrix(y_test).transpose())
									,1)    
        target = numpy.concatenate(((X_test_t),
									numpy.matrix(y_test_t).transpose()),1)
        source_yhat =  model.predict(X_test)
        target_yhat =  model.predict(X_test_t)
        source_proba = model.predict_proba(X_test)
        target_proba = model.predict_proba(X_test_t)

        SaveData(source=source,
		target=target,
        source_yhat=source_yhat,
		 target_yhat=target_yhat,
		source_proba=source_proba,
		target_proba=target_proba,
   	 source_path="temp/source.csv",
             target_path="temp/target.csv",
			 source_yhat_path="temp/source_yhat.csv",
			 target_yhat_path="temp/target_yhat.csv",
			 source_proba_path="temp/source_proba.csv",
			 target_proba_path="temp/target_proba.csv",)
			 
        return source, target, \
		source_yhat, target_yhat, \
		source_proba, target_proba 		
		
		
def SaveData(source,
			 target,
			 source_yhat,
			 target_yhat,
			 source_proba,
			 target_proba,
			 source_path="source.csv",
             target_path="target.csv",
			 source_yhat_path="source_yhat.csv",
			 target_yhat_path="target_yhat.csv",
			 source_proba_path="source_proba.csv",
			 target_proba_path="target_proba.csv",
			 ):	
    numpy.savetxt(source_path, source, delimiter=",",fmt='%d')
    numpy.savetxt(target_path,   target, delimiter=",",fmt='%d')		
    numpy.savetxt(source_yhat_path,   source_yhat, delimiter=",",fmt='%d')
    numpy.savetxt(target_yhat_path,   target_yhat, delimiter=",",fmt='%d')
    numpy.savetxt(source_proba_path,   source_proba, delimiter=",",fmt='%f')
    numpy.savetxt(target_proba_path,   target_proba, delimiter=",",fmt='%f')
        			 