#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:41:10 2018

@author: mlpa
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# import os
import csv

class Face_kNN_Classification:
    def __init__(self, w_final , train_data, test_data , mean_tr, std_tr):
        self.w_final = w_final
        self.train_data = train_data
        self.test_data = test_data
        self.mean_tr = mean_tr
        self.std_tr = std_tr


    def DATA_NORMALIZE(self, data, mean_f, std_f):
        print("DATA NORMALIZE INITIALIZE")
    
        N_Tr, N_F = data.shape
        for index in range(N_Tr):
            for jndex in range(N_F):
                if std_f[jndex] == 0:
                    data[index,jndex] = 0
                else:
                    data[index,jndex] = ((data[index,jndex] - mean_f[jndex]) / std_f[jndex])
            clear = lambda:os.system('clear') 
            clear() #function call
            
        return data
        
    def KNN_INIT(self, K_vec, k_fold):
        w_final = self.w_final
        gal_data = self.train_data
        probe_data = self.test_data
        
        K_vec = K_vec
        
        #if is_given == True:
        #    print("Use The Datasets Your Given Index Parameter and You will be Insert")
        #    #gal_data = self.LOAD_DATASETS_GIVEN(self.gal_data_path, start_idx, end_idx)
        #    #probe_data = self.LOAD_DATASETS_GIVEN(self.gal_data_path, start_idx, end_idx)
        #elif is_given == False:
        #    print("Given Train , Test datasets will be earned")
        #    gal_data = self.LOAD_DATASETS(self.gal_data_path)
        #    probe_data = self.LOAD_DATASETS(self.probe_data_path)
        
        #w_final = self.w_final
        #gal_data = self.loaded_data_train
        #probe_data = self.loaded_data_label
        mean_tr = self.mean_tr
        std_tr = self.std_tr
        #K_vec = self.K_vec
        out_file = "out_file_naming"
        print("out_file = ", out_file)
        
        gal_row, gal_col = gal_data.shape
        prb_row, prb_col = probe_data.shape
        
        #correct_rate_vec = {}

        data_gal = gal_data[:,:-1]
        class_gal = gal_data[:,gal_col-1]
        N_gal , N_F = data_gal.shape
        data_probe = probe_data[:,:-1]
        class_probe = probe_data[:,prb_col-1]
        N_probe, N_F_1 = data_probe.shape
        
        if N_F != N_F_1:
            print("data_Error")
            return print("EXIT")
        
        #sorted_data_gal, N_gal, N_f, N_C_gal, class_inform, N_class_sample_gal = self.DATA_SORT_INFORMATION(gal_data)
        
        gal_norm = self.DATA_NORMALIZE(data_gal, mean_tr, std_tr)
        probe_norm = self.DATA_NORMALIZE(data_probe, mean_tr, std_tr)
        
        gal_prj = np.dot(gal_norm, w_final)
        probe_prj = np.dot(probe_norm, w_final)
        
        neigh = KNeighborsClassifier(n_neighbors=K_vec, weights='distance')
        neigh.fit(gal_prj, class_gal)
        #probe_prj2 = probe_prj[:200]
        #neigh.fit(gal_prj, probe_prj2)
        #predic = neigh.predict(probe_prj)
        #predic_prob = neigh.predict_proba(probe_prj)
        predic_score = neigh.score(probe_prj, class_probe)
        #predic_score = neigh.score(probe_prj2,gal_prj)
        #print(predic)
        #print(predic_prob)
        print(predic_score)
        
        STR = "Result : [%d/10]Train Data(%d, %d), Test Data(%d, %d), K=%d , ACCURACY=%f" %(k_fold, gal_row, gal_col, prb_row, prb_col ,K_vec, predic_score)
        print(STR)
        return STR, predic_score
        

        
        
        
        
        