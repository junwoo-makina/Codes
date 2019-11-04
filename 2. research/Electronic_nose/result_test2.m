%% Train with L2PCA, L1PCA

gt_tr = table2array(trainiter1set1);
data_tr = table2array(trainiter1set1loss50trm);
data_te = table2array(testiter1set1loss50);
label_tr = table2array(testiter1set1labeltr);
label_te = table2array(testiter1set1labelte);
%dae_te = table2array(testiter1set1loss50reconDAE);

% L1PCA -> dimension reduction with L2PCA -> LDA ->evaluation
[W_L1PCA, n_it, elap_time] = L1PCA(gt_tr, 100);
reconstructed_tr = data_tr*W_L1PCA*W_L1PCA';
reconstructed_te = data_te*W_L1PCA*W_L1PCA';

[W_L2PCA, elpa_time2] = L2PCA_new(reconstructed_tr, 1, 100);

w_lda = LDA(gt_tr*W_L1PCA, label_tr);
L = [ones(20,1) reconstructed_te*W_L1PCA] * w_lda';
[~,P] = max(L');
label_test = P';

% quantify knn classifier
compressed_tr = data_tr*W_L2PCA;
compressed_te = data_te*W_L2PCA;
%compressed_tr = compressed_tr(:,2:end);
%compressed_te = compressed_te(:,2:end);
cknn = fitcknn(reconstructed_tr, label_tr, 'NumNeighbors', 5);
%rloss = resubLoss(cknn);
%CVmdl = crossval(cknn);
%kloss = kfoldLoss(CVmdl);

label_test_knn = predict(cknn, reconstructed_te);

precision = size(find(label_te == label_test),1)/20;
precision2 = size(find(label_test == label_test_knn),1)/20;
precision3 = size(find(label_te == label_test_knn),1)/20;
fprintf('\nmissing %d precision\n', precision);
fprintf('missing %d precision2\n', precision2);
fprintf('missing %d precision3\n', precision3);

%X1 = tsne(train_140);
%X2 = tsne(train_miss50_140);
%Y1 = tsne(reconstructed_tr);
%Y2 = tsne(reconstructed_te);

%figure
%gscatter(X1(:,1), X1(:,2));
%gscatter(X2(:,1), X2(:,2));
%gscatter(Y1(:,1), Y1(:,2));
%gscatter(Y2(:,1), Y2(:,2));