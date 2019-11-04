clc
clear all
close all
%%
for hnd = 40:10:90
    %total precision
    tot_precision = 0;
    for ind = 0:0
        %fprintf('execute %d/8 - fold data\n', ind);
        gt_tr = load(strcat('save',num2str(90),'/gt_tr_',num2str(ind),'.txt'));
        label_te = load(strcat('save',num2str(hnd),'/label_te_',num2str(ind),'.txt'));
        label_tr = load(strcat('save',num2str(hnd),'/label_tr_',num2str(ind),'.txt'));
        
        data_te = load(strcat('save',num2str(hnd),'/gt_tem_',num2str(ind),'.txt'));
        data_tr = load(strcat('save',num2str(hnd),'/gt_trm_',num2str(ind),'.txt'));
        
        %% vae
        
        vae_tr = load(strcat('save',num2str(hnd),'/reconstructed_trm_',num2str(ind),'.txt'));
        vae_te = load(strcat('save',num2str(hnd),'/reconstructed_tem_',num2str(ind),'.txt'));
        
        %% l1 pca
        
        [w, n_it, elap_time] = L1PCA(gt_tr, 100);
        
        l1pca_tr = (data_tr*w*w');
        l1pca_te = (data_te*w*w');
        
        %% pca
        
        [w, elap_time] =  L2PCA_new(gt_tr, 1, 100);
        
        l2pca_tr = (data_tr*w*w');
        l2pca_te = (data_te*w*w');
        
        
        
        %% mse
        mse_tr_vae = sum((vae_tr - gt_tr).*(vae_tr - gt_tr));
        mse_tr_l1pca = sum((l1pca_tr - gt_tr).*(l1pca_tr - gt_tr));
        mse_tr_l2pca = sum((l2pca_tr - gt_tr).*(l2pca_tr - gt_tr));
        mse_tr_gt = sum((data_tr - gt_tr).*(data_tr - gt_tr));
        
    end
    
    x = 1:1:32000;
    
    
    plot(x,mse_tr_vae, x, mse_tr_l1pca, x, mse_tr_l2pca, x, mse_tr_gt);
    
    
    fprintf('%d: %f %f %f\n', hnd, mean(mse_tr_vae), mean(mse_tr_l1pca), mean(mse_tr_l2pca));
    
    
end