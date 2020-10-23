classdef evalTuningParam
    properties
        type, threshval, compval, perfmat, perf_best, compval_best, threshval_best, perf_1se, compval_1se, threshval_1se;
    end
    methods
        function ParamTunObj = evalTuningParam(TPLScvmdl,type,X,Y,compvec,threshvec,subfold)
            assert(ismember(type,{'Pearson','Spearman','AUC'}),'performance metric must be one of ''Pearson'',''Spearman'',or ''AUC''')
            if nargin<7
                subfold = TPLScvmdl.testfold; % if no subdivision provided, same as CV fold
            end
            
            % Perform CV prediction and performance measurement
            threshvec = sort(threshvec); compvec = sort(compvec); % sorted from low to high
            perfmat = nan(length(compvec),length(threshvec),TPLScvmdl.numfold);
            for i = 1:TPLScvmdl.numfold
                disp(['Fold #',num2str(i)])
                testCVfold = TPLScvmdl.testfold == i;
                Ytest = Y(testCVfold);
                testsubfold = subfold(testCVfold);
                uniqtestsubfold = unique(testsubfold);
                for j = 1:length(threshvec)
                    predmat =  predict(TPLScvmdl.cvMdls{i},compvec,threshvec(j),X(testCVfold,:));
                    smallperfmat = nan(length(compvec),length(uniqtestsubfold));
                    for k = 1:length(uniqtestsubfold)
                        subfoldsel = testsubfold == uniqtestsubfold(k);
                        smallperfmat(:,k) = util_perfmetric(predmat(subfoldsel,:),Ytest(subfoldsel),type);
                    end
                    perfmat(:,j,i) = nanmean(smallperfmat,2);
                end
            end
            
            % prepare output object
            ParamTunObj.type = type; ParamTunObj.threshval = threshvec; ParamTunObj.compval = compvec; ParamTunObj.perfmat = perfmat;
            
            % find the point of maximum CV performance
            [ParamTunObj.perf_best,row_best,col_best,ParamTunObj.perf_1se,row_1se,col_1se] = findBestPerf(perfmat);
            ParamTunObj.compval_best = compvec(row_best); ParamTunObj.threshval_best = threshvec(col_best);
            ParamTunObj.compval_1se = compvec(row_1se); ParamTunObj.threshval_1se = threshvec(col_1se);
        end
        function plot(ParamTunObj)
            figure
            meansurf = nanmean(ParamTunObj.perfmat,3);
            [X,Y] = meshgrid(ParamTunObj.threshval,ParamTunObj.compval);
            surf(X,Y,meansurf,'EdgeColor',[.5,.5,.5],'FaceAlpha',0.5,'FaceColor','interp')
            ylabel('Number of PLS components'); xlabel('Proportion of Voxels Left'); zlabel(ParamTunObj.type)
            hold on
            [maxroute,ind] = max(mean(ParamTunObj.perfmat,3)); % finding the best map at each threshold point
            h0 = plot3(ParamTunObj.threshval,ParamTunObj.compval(ind),maxroute,'o-','MarkerSize',5,'MarkerFaceColor',[0.3,0.3,0.3]);
            [maxroute,ind] = max(mean(ParamTunObj.perfmat,3),[],2); % finding the best map at each component
            h1 = plot3(ParamTunObj.threshval(ind),ParamTunObj.compval,maxroute,'o-','MarkerSize',5,'MarkerFaceColor',[0.7,0.7,0.7]);
            h2 = plot3(ParamTunObj.threshval_best,ParamTunObj.compval_best-0.1,ParamTunObj.perf_best,'bo','MarkerSize',10,'MarkerFaceColor',[0.7,1,1]);
            h3 = plot3(ParamTunObj.threshval_1se,ParamTunObj.compval_1se+0.1,ParamTunObj.perf_1se,'ro','MarkerSize',10,'MarkerFaceColor',[1,1,0.7]);
            legend([h0,h1,h2,h3],{'best at threshold','best at component','Max Perf','1SE Perf'})
        end
    end
end

function [perf_best,row_best,col_best,perf_1se,row_1se,col_1se] = findBestPerf(perfmat)
avgperfmat = nanmean(perfmat,3); perf_best = max(avgperfmat(:));
[row_best,col_best] = find(avgperfmat==perf_best,1,'first');
standardError = nanstd(squeeze(perfmat(row_best,col_best,:)))/size(perfmat,3); % finding the standard error of the best point
candidates = avgperfmat(:,1:col_best)>(perf_best-standardError); % finding points whose metric is higher than perf_max minus 1 SE
[row_1se,col_1se] = find(candidates,1,'first');
perf_1se = avgperfmat(row_1se,col_1se);
end

function Perf = util_perfmetric(predmat,testY,type)
if strcmp(type,'AUC')
    n = size(testY,1); num_pos = sum(testY==1); num_neg = n - num_pos;
    if (num_pos>0 && num_pos < n)
        ranks = tiedrank(predmat); Perf = ( sum( ranks(testY==1,:) ) - num_pos * (num_pos+1)/2) / ( num_pos * num_neg);
    end
else
    Perf = corr(testY,predmat,'type',type);
end
end