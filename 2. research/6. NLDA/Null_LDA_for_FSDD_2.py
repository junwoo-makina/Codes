#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:39:43 2018

@author: mlpa
"""

import numpy as np
import pandas as pd
#import matlab.engine as matlab
#import scipy.linalg
import os

#eng = matlab.engine.start_matlab()

class Null_LDA:
    def __init__(self, out_file):
        self.out_file = out_file
    
    def LOAD_DATASETS(self, data_path):
        file = pd.read_csv(filepath_or_buffer=data_path,header=None,skiprows=0)
        print('Load Data, path is ::' , data_path)
        print('file Shape is : ',file.shape)
        file = np.array(file)
        file_row , file_col = file.shape
        self.tr_data = file
        self.label_data = file[:,file_col-1] # last index
        print("DATA Load Done !")
    
    def LOAD_DATASETS_without_path(self, data):
        file_row, file_col = data.shape
        self.tr_data = data
        self.label_data = data[:,file_col-1]

    def DATA_NORMALIZE(self, data, mean_f, std_f):

        N_Tr, N_F = data.shape
        for index in range(N_Tr):
            for jndex in range(N_F):
                if std_f[jndex] == 0:
                    data[index,jndex] = 0
                else:
                    data[index,jndex] = ((data[index,jndex] - mean_f[jndex]) / std_f[jndex])
            clear = lambda:os.system('clear') 
            clear() #function call
            #print(" Normalize : ",index)
            
        return data

    def CAL_STD(self, data):

        N_P , N_F = data.shape
        sum_dic = {}
        mean_vec = {}
        std_vec = {}

        for jndex in range(N_F):
            sum_dic[jndex] = 0
            for index in range(N_P):
                sum_dic[jndex] += data[index,jndex]
            mean_vec[jndex] = sum_dic[jndex] / N_P

        for jndex in range(N_F):
            sum_dic[jndex] = 0
            for index in range(N_P):
                sum_dic[jndex] = sum_dic[jndex] + (data[index,jndex] - mean_vec[jndex])*(data[index,jndex] - mean_vec[jndex])
            std_vec[jndex] = np.sqrt( ( sum_dic[jndex] ) / (N_P -1) )

        print("Calculation Standard and Variation is Done")
        return mean_vec , std_vec

    def CLASS_SEPARATION(self, data, label_data, class_label, N_class):
        # data , class , class_label, N_class in MATLAB
        print('===================================================================================')        
        print('\n ::: Class Separation Initialize :::')
        N_data , N_f = data.shape
        sep_class = np.array([])
        sep_dic = {}
        
        # 
        for index in range(N_class):
            #exec(['sep_class_' + 'str(class_label[index]))' + '= np.zeros(1, N_f)'])
            #print(str(class_1))
            sep_class = np.zeros((1,N_f))
            sep_dic[index] = sep_class

        #
        for index in range(N_data):
            sep_class = sep_dic[label_data[index ]]
            
            if np.sum(sep_class) == 0 :
                sep_class = data[index,:]
            else:
                sep_class = np.vstack((sep_class, data[index,:]))
            sep_dic[label_data[index]] = sep_class
            
        print('test 1 class: ',sep_dic[0].shape)
        print('test 2 class: ',sep_dic[1].shape)
        
        # pima dataset column name discard, If datasets have not any column name, Don't Use this Code
        for index in range(N_class):
            sep_class = sep_dic[index]
            sep_dic[index] = sep_class[:,:] # name column discarded
         
        
        # because We use Dictionary, don't need sorted. It already Done
        #sorted_data = {}
        
        #for index in range(N_class):
        #    sorted_data[index] = sep_dic[index]
            
        # pima dataset column name discard, If datasets have not any column name, Don't Use this Code
        #for index in range(N_class):
        #    sorted_class = sep_dic[index]
        #    sorted_data[index] = sorted_class[:,:] # name column discarded
        sorted_data = sep_dic # edit this
        
        class_index = np.zeros((N_class, 2))
        start = 0
        
        for index in range(N_class):
            N_sam , N_f = sep_dic[index].shape
            last = start + N_sam - 1
            class_index[index,0] = start
            class_index[index,1] = last
            start = last + 1

        print('Class Separation is Done !')
        
        return sorted_data , class_index     
    def CLASS_INFORMATION(self, file):
        # label_data is 'class' variable in MATLAB
        
        N_data = len(file)
        #print(N_data)
        
        N_class = 1
        #class_label = np.zeros(file.shape)
        class_label = list()
        class_label.append(int(file[0]))
        #print('N_class is :', N_class)
        #print('class_label (1) is :', int(class_label[0]))
        
        for index in range(N_data):
            if file[index] not in class_label:
                N_class += 1
                class_label.append(int(file[index]))

        class_label = sorted(class_label)
        #print(class_label)
        
        N_class_sample = {}
        
        for key in range(N_class):
            N_class_sample[key] = 0
        
        for key in N_class_sample:
            for index in range(N_data):
                if int(key) is int(file[index]):
                    N_class_sample[key] += 1
        
        self.N_class = N_class
        self.class_label = class_label
        self.N_class_sample = N_class_sample
        print('N_cFSDD_Pythonlass :',self.N_class)
        print('class_label :',self.class_label)
        print('N_class_sample :',self.N_class_sample)
        print('Class Information is Done !')
        return N_class,class_label,N_class_sample

    def DATA_SORT_INFORMATION(self, in_data):
        irow, icol = in_data.shape
        only_data = in_data[:, :icol-1]
        N_data, N_F = only_data.shape

        label_data = in_data[:,icol-1] # label_data is 'class' variable in MATLAB
        N_C, class_label, N_class_sample = self.CLASS_INFORMATION(label_data)
        sorted_data, class_index = self.CLASS_SEPARATION(in_data, label_data, class_label, N_C)

        # class_infrom, tem = sort(class_label') MATLAB CODE
        class_label = np.transpose(class_label)
        sorted_class_label = sorted(class_label)
        class_inform = sorted_class_label
        
        true_data_tr = sorted_data[0]
        for key in sorted_data:
            if true_data_tr.shape == sorted_data[key].shape:
                continue
            else:
                true_data_tr = np.vstack((true_data_tr , sorted_data[key]))

        
        return true_data_tr, N_data, N_F, N_C, class_inform, N_class_sample

    def NULL_LDA(self):
        print("NULL_LDA Initialize")
        
        sorted_data, N_Tr, N_F, N_C, class_inform, N_class_sample = self.DATA_SORT_INFORMATION(self.tr_data)
        print(N_F, N_C, class_inform, N_class_sample)
        
        srow, scol = sorted_data.shape
        data_tr = sorted_data[:,:scol-1]
        class_tr = sorted_data[:,scol-1]

        mean_tr, std_tr = self.CAL_STD(data_tr)
        data_tr = self.DATA_NORMALIZE(data_tr, mean_tr, std_tr)
        print("DATA Normalize Done")

        class_0 = data_tr[:N_class_sample[0], :]
        start_idx = N_class_sample[0]
        for index in range(1,N_C):
            end_idx = start_idx + N_class_sample[index]
            temp_str1 = 'class_' + str(index) + ' = data_tr[start_idx: end_idx]'
            exec(temp_str1)
            start_idx = start_idx + N_class_sample[index]

        self.one_tr_data_per_class = np.zeros((N_C,scol-1))
        
        for index in range(N_C):
            print("[ %d / %d ] get one_tr_data_per_class Count." %(index, N_C), end="\r")
            if ((N_C - index) == 0):
                print("[ %d / %d ] get one_tr_data_per_class Count." %(index + 1, N_C))
            temp_str2 = 'trow, tcol = class_' + str(index) + '.shape'
            exec(temp_str2)
            temp_str3 = 'self.one_tr_data_per_class['+ str(index) +', : ] = class_' + str(index) + '[trow-1, :]'
            exec(temp_str3)
        
        one_tr_data_per_class = self.one_tr_data_per_class

        sum_class = np.zeros((N_C, N_F))
        mean_class = np.zeros((N_C, N_F))

        for index in range(N_C):
            print("[ %d / %d ] N_C Count." %(index, N_C), end="\r")
            for jndex in range(N_class_sample[index]):
                temp_str4 = 'sum_class[index,:] = sum_class[index, :] + class_' + str(index) + '[jndex,:]'
                exec(temp_str4)
            mean_class[index, :] = sum_class[index, :] / N_class_sample[index]
            
        #t0 = 0

        A = np.zeros([0,N_F])
        #self.data_rest = np.array([])
        for index in range(N_C):
            for jndex in range(N_class_sample[index]):
                temp_str5 = 'self.data_rest = (class_' + str(index) + '[jndex, :] - mean_class[index, :])'
                exec(temp_str5)
                T_data_rest = np.transpose(self.data_rest)
                A = np.vstack((A, T_data_rest))

        T_A = np.transpose(A)
        K = (np.dot(A,T_A)) / N_Tr

        ra_K = np.linalg.matrix_rank(K)
        print("exec Section Done")
        print("matirx Rank : " ,ra_K)
        
        # matlab = E, D
        # python = w , v
        #q_comp, d = scipy.linalg.eig(K)
        
        K = np.array(K)
        
        #td, tq_comp = matlab.eig(K)
        print("First Eigen Decomposition")
        
        td, tq_comp = np.linalg.eig(K)
        #d = np.diag(d)
        #q_comp = np.real_if_close(tq_comp)
        #d = np.real_if_close(np.diag(td)) 
        q_comp = tq_comp
        d = np.diag(td)
        
        #np.save('/home/mlpa/Documents/mlpa/FSDD_Python/Eigen_Similarity/q_comp_py.npy', q_comp)
        #np.save('/home/mlpa/Documents/mlpa/FSDD_Python/Eigen_Similarity/d_py.npy', d)
        #d = np.real_if_close(np.diag(d))
        
        d_temp = list()
        w = np.zeros((N_Tr, 0))
        #q = np.array(([]))
        for index in range(N_Tr):
            d_temp.append(d[index, index])
        print("got d_temp arrays")
        
        eig_val = sorted(d_temp)
        d_idx = list()
        for i in range(len(eig_val)):
            d_idx.append(i)
        #d_idx = sorted(d_temp)
        
        w_range = np.zeros((N_F, ra_K))
        for index in range(N_Tr):
            temp =  np.array(q_comp[:, int(d_idx[(N_Tr-1) -index])])
            #temp = np.transpose(temp)
            w = np.column_stack((w, temp))
        print("got w matrix")
        print("Prevent Discard Imaginary parts")
        q = np.zeros((N_F, N_Tr))
        q = q.astype('complex')
            
        for index in range(N_Tr):
            temp_q = T_A @ w[:,index]
            q[:,index] = temp_q.astype('complex')
            norm_weight = np.linalg.norm(q[:,index])
            q[:,index] = q[:,index] / norm_weight
            
            
        w_range = q[:,:ra_K]
        

        T_w_range = np.transpose(w_range)
        T_one_tr_data_per_class = np.transpose(one_tr_data_per_class)
        temp10 = T_w_range @ T_one_tr_data_per_class
        #x_range = np.matmul(w_range , temp10)
        x_range = w_range @ temp10
        x_com = T_one_tr_data_per_class - x_range
        T_x_com = np.transpose(x_com)
        mean_x_com = np.transpose(np.mean(T_x_com))
        
        A_com = np.zeros((N_F,))
        for index in range(N_C):
            temp_sqrt = np.sqrt(N_class_sample[index]) * N_Tr
            temp11 = ( x_com[:, index] - mean_x_com ) * temp_sqrt
            A_com = np.column_stack((A_com, temp11))
        
        A_com = A_com[:,1:]
        #K_com = (np.transpose(A_com) * A_com) / N_C
        #K_com = np.dot(np.transpose(A_com), A_com) / N_C
        K_com = ( np.transpose(A_com) @ A_com ) / N_C
        ra_K_com = np.linalg.matrix_rank(K_com)
        
        d_temp_com = list()
        print("Second Eigen Decomposition")
        d_com, q_comp_com  = np.linalg.eig(K_com)
        #d_com = np.diag(d_com)
        #d_com = np.real_if_close(np.diag(d_com))      
        d_com = np.diag(d_com)
        #np.save('/home/mlpa/Documents/mlpa/FSDD_Python/Eigen_Similarity/q_comp_com_py.npy', q_comp_com)
        #np.save('/home/mlpa/Documents/mlpa/FSDD_Python/Eigen_Similarity/d_com_py.npy', d_com)
        
        for index in range(N_C):
            d_temp_com.append(d_com[index, index])
        
        
        eig_val_com = sorted(d_temp_com)
        d_idx_com = list()
        for i in range(len(eig_val_com)):
            d_idx_com.append(i)
        
        #for index in range(len(d_temp_com)):
        #    d_idx_com[index] = index
        
        w_dis = np.zeros((N_C,N_C))
        for index in range(N_C):
            w_dis[:,index] = q_comp_com[:, d_idx_com[N_C -index - 1]]
        
        w_dis_com = np.zeros((N_F ,N_C))
        print(N_C)
        for index in range(N_C):
            #w_dis_com[:,index] = np.dot(A_com , w_dis[:,index])
            w_dis_com[:,index] = A_com @ w_dis[:, index]
            norm_weight_com = np.linalg.norm(w_dis_com[:,index])
            w_dis_com[:,index] = w_dis_com[:,index] / norm_weight_com
            
            #Result = w_dis_com[:,index] / norm_weight_com
            #np.save('/home/mlpa/Documents/mlpa/FSDD_Python/Eigen_Similarity/NLDA_Result_py.npy',Result)
            # Result is ' W_dis_com ' 
            
        W_dis_com = w_dis_com[:, :ra_K_com]
        Result = W_dis_com
        self.w_dis_com = w_dis_com
        self.std_tr = std_tr
        self.mean_tr = mean_tr
            
        return Result, std_tr, mean_tr
    
    
#data_path = '/home/mlpa/Documents/mlpa/Python_WorkSpace/FSDD_NLDA_Python/pima.csv'
data_path = '/home/mlpa/Documents/mlpa/CNN/data_result/mono/flatten.csv'
##data_path = /home/mlpa/Documents/Dku_Cau_CROSS/datasets'
#
NLDA = Null_LDA("PIMA_NLDA_OUTPUT")
NLDA.LOAD_DATASETS(data_path)
result, std_tr, mean_tr = NLDA.NULL_LDA()
print(result, std_tr, mean_tr)
#
#
#
#











