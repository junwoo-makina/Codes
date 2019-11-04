%% Train with L2PCA, L1PCA

train_140 = table2array(trainiter6set6);
test_20 = table2array(testiter6set6loss50);
tr_label = table2array(testiter6set6labeltr);
te_label = table2array(testiter6set6labelte);
[W_L2PCA, elap_time] = L2PCA_new(train_140,1, 100);
[W_L1PCA, elap_time2] = L1PCA(train_140, 100);

% recon l2pca, l1pca, DAE
recon_L2PCA = train_140*W_L2PCA*W_L2PCA';
recon_L1PCA = train_140*W_L1PCA*W_L1PCA';

% Test with LDA
w_lda_L2PCA = LDA(train_140*W_L2PCA, tr_label);
w_lda_L1PCA = LDA(train_140*W_L1PCA, tr_label);

% train with original LDA
L_L2PCA = [ones(20,1) recon_L2PCA*W_L2PCA] * w_lda_L2PCA';
L_L1PCA = [ones(20,1) recon_L1PCA*W_L1PCA] * w_lda_L1PCA';

[~, P_L2PCA] = max(L_L2PCA');
label_test_L2PCA = P_L2PCA';
[~, P_L1PCA] = max(L_L1PCA');
label_test_L1PCA = P_L1PCA';

%% Quantify knn classifier
compressed_tr_L2PCA = train_140*W_L2PCA;
compressed_tr_L1PCA = train_140*W_L1PCA;
compressed_te_L2PCA = test_20*W_L2PCA;
compressed_te_L1PCA = test_20*W_L1PCA;


%compressed_tr_L2PCA = compressed_tr_L2PCA(:,2:end);
%compressed_tr_L1PCA = compressed_tr_L1PCA(:,2:end);
%compressed_te_L2PCA = compressed_te_L2PCA(:,2:end);
%compressed_te_L1PCA = compressed_te_L1PCA(:,2:end);

cknn_L2PCA = fitcknn(compressed_tr_L2PCA, tr_label, 'NumNeighbors', 5);
rloss_L2PCA = resubLoss(cknn_L2PCA);
CVmdl_L2PCA = crossval(cknn_L2PCA);
kloss_L2PCA = kfoldLoss(CVmdl_L2PCA);

cknn_L1PCA = fitcknn(compressed_tr_L1PCA, tr_label, 'NumNeighbors', 5);
rloss_L1PCA = resubLoss(cknn_L1PCA);
CVmdl_L1PCA = crossval(cknn_L1PCA);
kloss_L1PCA = kfoldLoss(CVmdl_L1PCA);

% do predict
compressed_te_DAE = table2array(test_iter6_set6_loss50latent_trm);
label_test_knn_L2PCA = predict(cknn_L2PCA, compressed_te_L2PCA);
label_test_knn_L1PCA = predict(cknn_L1PCA, compressed_te_L1PCA);

precision = size(find(te_label == label_test_knn_L2PCA),1)/20;
precision2 = size(find(te_label == label_test_knn_L1PCA),1)/20;

fprintf('L2PCA accuracy : %d\n', precision);
fprintf('L1PCA accuracy : %d\n', precision2);