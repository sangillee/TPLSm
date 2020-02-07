function Perf = util_perfmetric(predmat,testY,type)
switch type
    case 'AUC'
        predmat(predmat>1) = 1; predmat(predmat<0) = 0; % capping
        Perf = localAUC(testY==1,predmat); % AUC
    case 'BACC'
        Perf = localbACC(testY==1,predmat>0.5); % balanced Accuracy
    case 'ACC'
        Perf = mean(bsxfun(@eq,(testY==1),(predmat>0.5))); % accuracy
    case 'PEARSON'
        Perf = corr(testY,predmat);
    case 'SPEARMAN'
        Perf = corr(testY,predmat,'type','Spearman');
    case 'MSE'
        resid = bsxfun(@minus,testY,predmat);
        Perf = mean(resid.^2);
    case 'RMSE'
        resid = bsxfun(@minus,testY,predmat);
        Perf = sqrt(mean(resid.^2));
    case 'MAD'
        resid = bsxfun(@minus,testY,predmat);
        Perf = mean(abs(resid));
    otherwise
        error('unknown performance measure type')
end
end

function auc = localAUC(truth,score) % area under receiver operating characteristic curve
n = size(truth,1); num_pos = sum(truth); num_neg = n - num_pos;
auc = nan(1,size(score,2));
if (num_pos>0 && num_pos < n)
    ranks = tiedrank(score);
    auc = ( sum( ranks(truth,:) ) - num_pos * (num_pos+1)/2) / ( num_pos * num_neg);
end
end

function acc = localbACC(truth,prediction) % balanced Accuracy
truth = repmat(truth,1,size(prediction,2));
acc1 = sum(prediction==1 & truth==1)./sum(truth==1);
acc2 = sum(prediction==0 & truth==0)./sum(truth==0);
acc = (acc1+acc2)/2;
end