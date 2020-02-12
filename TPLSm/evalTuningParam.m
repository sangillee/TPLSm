classdef evalTuningParam
    properties
        type, threshval, compval, perfmat, perf_max, compval_max, threshval_max, SE, perf_1se, compval_1se, threshval_1se;
    end
    methods
        function ParamTunObj = evalTuningParam(TPLScvmdl,type,X,Y,compvec,threshvec)
            % Perform CV prediction and performance measurement
            perfmat = nan(length(compvec),length(threshvec),TPLScvmdl.numfold);
            for i = 1:TPLScvmdl.numfold
                disp(['Fold #',num2str(i)])
                test = TPLScvmdl.testfold == i;
                for j = 1:length(threshvec)
                    predmat =  predict(TPLScvmdl.cvMdls{i},compvec,threshvec(j),X(test,:));
                    perfmat(:,j,i) = util_perfmetric(predmat,Y(test),type);
                end
            end
            avgperfmat = mean(perfmat,3);
            
            % prepare output object
            ParamTunObj.type = type; % specify tuning performance type
            ParamTunObj.threshval = threshvec;
            ParamTunObj.compval = compvec;
            ParamTunObj.perfmat = perfmat;
            
            % find the point of maximum CV performance
            if ismember(type,{'MSE','RMSE','MAD'}) % for these metrics, lower value is better
                ParamTunObj.perf_max = min(avgperfmat(:));
            else
                ParamTunObj.perf_max = max(avgperfmat(:));
            end
            [row,col] = find(avgperfmat==ParamTunObj.perf_max);
            ParamTunObj.compval_max = compvec(row);
            ParamTunObj.threshval_max = threshvec(col);
            
            % find the most parsimonious model (lower threshval) that is within 1 SE of maximum CV point
            ParamTunObj.SE = std(squeeze(perfmat(row,col,:)))/sqrt(TPLScvmdl.numfold);
            if ismember(type,{'MSE','RMSE','MAD'}) % for these metrics, lower value is better
                candidates = avgperfmat(:,1:col)<(ParamTunObj.perf_max+ParamTunObj.SE); % finding points whose metric is lower than perf_max plus 1 SE
            else
                candidates = avgperfmat(:,1:col)>(ParamTunObj.perf_max-ParamTunObj.SE); % finding points whose metric is higher than perf_max minus 1 SE
            end
            [row,col] = find(candidates,1,'first');
            ParamTunObj.perf_1se = avgperfmat(row,col);
            ParamTunObj.compval_1se = compvec(row);
            ParamTunObj.threshval_1se = threshvec(col);
        end
        function plot(ParamTunObj)
            figure
            [X,Y] = meshgrid(ParamTunObj.threshval,ParamTunObj.compval);
            surf(X,Y,mean(ParamTunObj.perfmat,3),'EdgeColor',[.5,.5,.5],'FaceAlpha',0.5,'FaceColor','interp')
            ylabel('Number of PLS components'); xlabel('Proportion of Voxels Left'); zlabel(ParamTunObj.type)
            set(gca, 'XScale', 'log')
            hold on
            h1 = plot3(ParamTunObj.threshval_max,ParamTunObj.compval_max,ParamTunObj.perf_max,'bo','MarkerSize',10,'MarkerFaceColor',[0.7,1,1]);
            h2 = plot3(ParamTunObj.threshval_1se,ParamTunObj.compval_1se,ParamTunObj.perf_1se,'ro','MarkerSize',10,'MarkerFaceColor',[1,1,0.7]);
            legend([h1,h2],{'Max Perf','1SE Perf'})
        end
    end
end

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