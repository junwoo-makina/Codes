
..clc
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
            recon_te = load(strcat('iter',num2str(iter),'/set',num2str(set),'/test_iter',num2str(iter),'_set',num2str(set),'_loss',num2str(miss),'_recon_DAE.txt'));
            
			
            % Reconstruct L1, L2 PCA with original and missing data
            [W_L1PCA, n_it, elap_time] = L1PCA(data_tr, 100);
            [W_L2PCA, elap_time2] = L2PCA_new(data_tr, 1, 100);
            w_lda_L1PCA = LDA(data_tr*W_L1PCA, label_tr);
            w_lda_L2PCA = LDA(data_tr*W_L2PCA, label_tr);
            
            L_L1PCA = [ones(20,1) data_te*W_L1PCA*W_L1PCA'*W_L1PCA] * w_lda_L1PCA';
            [~,P_L1] = max(L_L1PCA');
            label_test_L1_LDA = P_L1';
            L_L2PCA = [ones(20,1) data_te*W_L2PCA*W_L2PCA'*W_L2PCA] * w_lda_L2PCA';
            [~,P_L2] = max(L_L2PCA');
            label_test_L2_LDA = P_L2';
            
            cknn_L1 = fitcknn(data_tr, label_tr, 'NumNeighbors', 5);
            cknn_L2 = fitcknn(data_tr, label_tr, 'NumNeighbors', 5);
            cknn_ae = fitcknn(data_tr, label_tr, 'Numneighbors', 5);
            
            label_test_knn_L1_R = predict(cknn_L1, data_te*W_L1PCA*W_L1PCA');
            label_test_knn_L1_M = predict(cknn_L1, data_te);
            label_test_knn_L2_R = predict(cknn_L2, data_te*W_L2PCA*W_L2PCA');
            label_test_knn_L2_M = predict(cknn_L2, data_te);
            label_test_knn_ae_R = predict(cknn_ae, recon_te);
            label_test_knn_ae_M = predict(cknn_ae, data_te);
            
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
       
       fprintf('\n for miss %d - %d-iter %d-set precision L1PCA with recon data : %d\n', miss, iter, set, iter_precision_L1);
       fprintf('\n for miss %d - %d-iter %d-set precision L1PCA with missing data : %d\n', miss, iter, set, iter_precision_L1_m);
       fprintf('\n for miss %d - %d-iter %d-set precision L2PCA with recon data : %d\n', miss, iter, set, iter_precision_L2);
       fprintf('\n for miss %d - %d-iter %d-set precision L2PCA with missing data : %d\n', miss, iter, set, iter_precision_L2_m);
       fprintf('\n for miss %d - %d-iter %d-set precision autoencoder with recon data : %d\n', miss, iter, set, iter_precision_ae);
       fprintf('\n for miss %d - %d-iter %d-set precision autoencoder with missing data : %d\n\n\n', miss, iter, set, iter_precision_ae_m);
    end
end