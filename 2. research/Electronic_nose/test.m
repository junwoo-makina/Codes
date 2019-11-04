clc
clear all
close all
%%
for hnd = 10:5:50
    %total precision  
    tot_precision = 0;   
    for ind = 0:7
        %fprintf('execute %d/8 - fold data\n', ind);
        gt_tr = load(strcat('save',num2str(50),'/gt_tr_',num2str(ind),'.txt'));
        label_te = load(strcat('save',num2str(hnd),'/label_te_',num2str(ind),'.txt'));
        label_tr = load(strcat('save',num2str(hnd),'/label_tr_',num2str(ind),'.txt'));
        
        data_te = load(strcat('save',num2str(hnd),'/gt_te_',num2str(ind), '.txt'));
        data_tr = load(strcat('save',num2str(hnd),'/gt_tr_',num2str(ind), '.txt'));
        data_tem = load(strcat('save',num2str(hnd),'/gt_tem_',num2str(ind),'.txt'));
        data_trm = load(strcat('save',num2str(hnd),'/gt_trm_',num2str(ind),'.txt'));
        
        %% reconstruction option
        %1. pca
        [w, elap_time] = L2PCA_new(gt_tr, 1, 100);
        %[w, n_it, elap_time] = L1PCA(gt_tr, 100);
                
        reconstructed_tr = (data_trm*w*w');
        reconstructed_te = (data_tem*w*w');
        
        %dimensionality reduction using pca 
        %[w, elap_time] = L2PCA_new(reconstructed_tr, 1, 100);
        %[w, n_it, elap_time] = L1PCA(gt_tr, 100);
        
        %%% test with LDA   
        %w_lda = LDA(reconstructed_tr*w, label_tr);
        %%[tr, te] = CLDA2(reconstructed_tr, reconstructed_te, label_tr, 120);
        
        %%train with original LDA
        %L = [ones(20,1) reconstructed_te*w] * w_lda';
        
        %[~,P] = max(L');
        %label_test = P';
        
        % quantify knn classifier
        compressed_tr = (data_trm*w*w')*w;
        compressed_te = (data_tem*w*w')*w;
        %compressed_tr = (data_tr)*w;
        %compressed_te = (data_te)*w;
        
        %compressed_tr = reshape(double(tr), [200, 140])';
        %compressed_te = reshape(double(te), [200, 20])';
        
             compressed_tr = compressed_tr(:,2:end);
             compressed_te = compressed_te(:,2:end);
        
        cknn = fitcknn(compressed_te, label_te, 'NumNeighbors', 5);
        rloss = resubLoss(cknn);
        CVmdl = crossval(cknn);
        kloss = kfoldLoss(CVmdl);
        
        %% do predict
        label_test = predict(cknn, compressed_te);
        
        precision = size(find(label_te == label_test),1)/20;
        
        tot_precision= tot_precision+precision;
        
        clear data_te; clear data_tr; clear label_te; clear label_tr;
        
        
        
    end
    
    tot_precision = tot_precision/8;
    
    fprintf('missing %d precision %f\n', hnd, tot_precision);
end