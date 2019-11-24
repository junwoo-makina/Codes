clc
clear all
close all

for miss = 10:5:50
    for iter = 1:1:10
        iter_precision_L1 = 0;                                                          % Reconstructed data
        iter_precision_L2 = 0;
        iter_precision_ae = 0;
        iter_precision_L1_m = 0;                                                        % Missing data
        iter_precision_L2_m = 0;
        iter_precision_ae_m = 0;
        for set = 1:1:8
            % load original data
            gt_tr = load(strcat('iter',num2str(iter),'/set',num2str(set),'/train_iter',num2str(iter),'_set',num2str(set),'.txt'));
            gt_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'.txt'));
            label_tr = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'label_tr.txt'));
            label_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'label_te.txt'));
            
            data_tr = load(strcat('iter',num2str(iter),'/set',num2str(set),'/train_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'trm.txt'));
            data_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'.txt'));
            latent_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'latent_te.txt'));
            latent_tem = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'latent_tem.txt'));
            latent_tr = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'latent_tr.txt'));
			
            % Reconstruct L1, L2 PCA with original and missing data
            [W_L1PCA, n_it, elap_time] = L1PCA(gt_tr, 100);
            [W_L2PCA, elap_time2] = L2PCA_new(gt_tr, 100);
            w_lda_L1PCA = LDA(gt_tr*W_L1PCA, label_tr);
            w_lda_L2PCA = LDA(gt_tr*W_L2PCA, label_tr);
            
            L_L1PCA = [ones(20,1) data_te*W_L1PCA*W_L1PCA'*W_L1PCA] * w_lda_L1PCA';
            [~,P_L1] = max(L_L1PCA');
            label_test_L1_LDA = P_L1';
            L_L2PCA = [ones(20,1) data_te*W_L2PCA*W_L2PCA'*W_L2PCA] * w_lda_L2PCA';
            [~,P_L2] = max(L_L2PCA');
            label_test_L2_LDA = P_L2';
            
            cknn_L1 = fitcknn(gt_tr*W_L1PCA, label_tr, 'NumNeighbors', 5);
            cknn_L2 = fitcknn(gt_tr*W_L1PCA, label_tr, 'NumNeighbors', 5);
            cknn_ae = fitcknn(latent_tr, label_tr, 'Numneighbors', 5);
            
            label_test_knn_L1_R = predict(cknn_L1, data_te*W_L1PCA*W_L1PCA'*W_L1PCA);
            label_test_knn_L1_M = predict(cknn_L1, data_te*W_L1PCA);
            label_test_knn_L2_R = predict(cknn_L2, data_te*W_L2PCA*W_L2PCA'*W_L2PCA);
            label_test_knn_L2_M = predict(cknn_L2, data_te_W_L2PCA);
            label_test_knn_ae_R = predict(cknn_ae, latent_te);
            label_test_knn_ae_M = predict(cknn_ae, latent_tem);
            
            precision_L1_R = size(find(label_te == label_test_knn_L1_R),1)/20;
            precision_L1_M = size(find(label_te == label_test_knn_L1_M),1)/20;
            precision_L2_R = size(find(label_te == label_test_knn_L2_R),1)/20;
            precision_L2_M = size(find(label_te == label_test_knn_L2_M),1)/20;
            precision_ae_R = size(find(label_te == label_test_knn_L2_R),1)/20;
            precision_ae_M = size(find(label_te == label_test_knn_L2_M),1)/20;
            
            iter_precision_L1 = iter_precision_L1 + precision_L1_R;
            iter_precision_L1_m = iter_precision_L1_m + precision_L1_M;
            iter_precision_L2 = iter_precision_L2 + precision_L2_R;
            iter_precision_L2_m = iter_precision_L2_m + precision_L2_M;
            iter_precision_ae = iter_precision_ae + precision_ae_R;
            iter_precision_ae_m = iter_precision_ae_m + precision_ae_M;
        end
       iter_precision_L1 = iter_precision_L1/8;
       iter_precision_L1_m = iter_precision_L1_m/8;
       iter_precision_L2 = iter_precision_L2/8;
       iter_precision_L2_m = iter_precision_L2_m/8;
       iter_precision_ae = iter_precision_ae/8;
       iter_precision_ae_m = iter_precision_ae_m/8;
       
       fprintf('\n%d-iter %d-set precision L1PCA with recon data : %d\n', iter, set, iter_precision_L1);
       fprintf('\n%d-iter %d-set precision L1PCA with missing data : %d\n', iter, set, iter_precision_L1_m);
       fprintf('\n%d-iter %d-set precision L2PCA with recon data : %d\n', iter, set, iter_precision_L2);
       fprintf('\n%d-iter %d-set precision L2PCA with missing data : %d\n', iter, set, iter_precision_L2_m);
       fprintf('\n%d-iter %d-set precision autoencoder with recon data : %d\n', iter, set, iter_precision_ae);
       fprintf('\n%d-iter %d-set precision autoencoder with missing data : %d\n\n\n', iter, set, iter_precision_ae_m);
    end
end

for iter = 1:1:10       % For 10 iterations
    iter_precision_L1 = 0;
    iter_precision_L2 = 0;
    iter_precision_knn = 0;
    iter_precision_knn_m = 0;
    for set = 1:1:8     % For 8 Sets(These are in the same experiments)
        label_tr = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'label_tr.txt'));
        label_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'label_te.txt'));
        
        % load original data
        gt_tr = load(strcat('iter',num2str(iter),'/set',num2str(set),'/train_iter',num2str(iter),'_set',num2str(set),'.txt'));
        gt_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'.txt'));
        
        % load missing data
        miss = 10;
        data_tr = load(strcat('iter',num2str(iter),'/set',num2str(set),'/train_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'trm.txt'));
        data_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'.txt'));
        latent_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'latent_te.txt'));
        latent_tem = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'latent_tem.txt'));
        latent_tr = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'latent_tr.txt'));
				
				% Reconstruct L1, L2 PCA with original and missing data
        [W_L1PCA, n_it, elap_time] = L1PCA(data_tr, 100);
        [W_L2PCA, elpa_time2] = L2PCA_new(data_tr, 1, 100);
        
        reconstructed_missing_te_L1 = data_te*W_L1PCA*W_L1PCA';
        reconstructed_missing_te_L2 = data_te*W_L2PCA*W_L2PCA';
		reconstructed_gt_te_L1 = gt_te*W_L1PCA*W_L1PCA';
		reconstructed_gt_te_L2 = gt_te*W_L2PCA*W_L2PCA';
        
        w_lda_L1PCA = LDA(gt_tr*W_L1PCA, label_tr);
        w_lda_L2PCA = LDA(gt_tr*W_L2PCA, label_tr);
        
        L_L1PCA = [ones(20,1) reconstructed_missing_te_L1*W_L1PCA] * w_lda_L1PCA';
        [~,P_L1] = max(L_L1PCA');
        label_test_L1_LDA = P_L1';
        L_L2PCA = [ones(20,1) reconstructed_missing_te_L2*W_L2PCA] * w_lda_L2PCA';
        [~,P_L2] = max(L_L2PCA');
        label_test_L2_LDA = P_L2';
        
        
        cknn = fitcknn(gt_tr*W_L2PCA, label_tr, 'NumNeighbors', 5);
        cknn2 = fitcknn(latent_tr, label_tr, 'NumNeighbors', 5);
        % For L1, L2
        label_test_knn_m = predict(cknn, data_te*W_L2PCA);                      % missing data
        label_test_knn = predict(cknn, reconstructed_missing_te_L1*W_L2PCA);    % reconstructed missing data
        % For DAE
        label_test_ae = predict(cknn2, latent_te);                              % reconstructed missing data
        label_test_ae_m = predict(cknn2, latent_tem);                           % missing data
        
        % For L1, L2
        precision_knn = size(find(label_te == label_test_knn),1)/20;
        precision_knn_m = size(find(label_te == label_test_knn_m),1)/20;
        % For DAE
        precision_L1 = size(find(label_te == label_test_ae),1)/20;
        precision_L2 = size(find(label_te == label_test_ae_m),1)/20;
            
        iter_precision_L1 = iter_precision_L1 + precision_L1;
        iter_precision_L2 = iter_precision_L2 + precision_L2;
        iter_precision_knn = iter_precision_knn + precision_knn;
        iter_precision_knn_m = iter_precision_knn_m + precision_knn_m;
           
        clear data_tr; clear data_te;
        
    end
    iter_precision_L1 = iter_precision_L1/8;
    iter_precision_L2 = iter_precision_L2/8;
    iter_precision_knn = iter_precision_knn/8;
    iter_precision_knn_m = iter_precision_knn_m/8;
    fprintf('\n%d-iter %d-set precision_ae : %d\n', iter, set, iter_precision_L1);
    fprintf('\n%d-iter %d-set precision_ae_M : %d\n', iter, set, iter_precision_L2);
    fprintf('\n%d-iter %d-set precision_knn : %d\n', iter, set, iter_precision_knn);
    fprintf('\n%d-iter %d-set preicision_knn_m : %d\n', iter, set, iter_precision_knn_m);
    fprintf('\n\n');
end
