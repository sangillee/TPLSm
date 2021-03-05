classdef evalTuningParam
    properties
        type, threshval, compval, perfmat, perf_best, compval_best, threshval_best, perf_1se, compval_1se, threshval_1se, best_at_threshold;
    end
    methods
        function ParamTunObj = evalTuningParam(cvmdl,type,X,Y,compvec,threshvec,subfold)
            % Evaluating cross-validation performance of a TPLS_cv model at compvec and threshvec
            %   'cvmdl'     : A TPLS_cv object
            %   'type'      : CV performance metric type. One of Pearson, Spearman, AUC, ACC, negMSE, negRMSE.
            %   'X'         : The same X as used in TPLS_cv.
            %   'Y'         : The same Y as used in TPLS_cv.
            %   'compvec'   : Vector of number of components to test in cross-validation.
            %   'threshvec' : Vector of threshold level [0 1] to test in cross-validation.
            %   'subfold'   : (Optional) Subdivision within testing fold to calculate performance. For example scan run division within subject.
            
            % input checking
            if nargin<7
                subfold = cvmdl.CVfold; % if no subdivision provided, same as CV fold
            end
            [compvec,threshvec] = evalTuningParaminputchecking(cvmdl,type,X,Y,compvec,threshvec,subfold);
            
            % Perform CV prediction and performance measurement
            perfmat = nan(length(compvec),length(threshvec),cvmdl.numfold);
            for i = 1:cvmdl.numfold
                disp(['Fold #',num2str(i)])
                testCVfold = cvmdl.CVfold(:,i) == 1;
                Ytest = Y(testCVfold);
                testsubfold = subfold(testCVfold);
                uniqtestsubfold = unique(testsubfold);
                for j = 1:length(threshvec)
                    predmat =  predict(cvmdl.cvMdls{i},compvec,threshvec(j),X(testCVfold,:));
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
            [ParamTunObj.perf_best,row_best,col_best,ParamTunObj.perf_1se,row_1se,col_1se,maxroute,maxrouteind] = findBestPerf(perfmat);
            ParamTunObj.compval_best = compvec(row_best); ParamTunObj.threshval_best = threshvec(col_best);
            ParamTunObj.compval_1se = compvec(row_1se); ParamTunObj.threshval_1se = threshvec(col_1se);
            ParamTunObj.best_at_threshold = [maxroute(:),threshvec,compvec(maxrouteind)];
        end
        function plot(ParamTunObj)
            meansurf = nanmean(ParamTunObj.perfmat,3);
            [X,Y] = meshgrid(ParamTunObj.threshval,ParamTunObj.compval);
            surf(X,Y,meansurf,'EdgeColor',[.5,.5,.5],'FaceAlpha',0.5,'FaceColor','interp')
            ylabel('Number of PLS components'); xlabel('Proportion of Voxels Left'); zlabel(ParamTunObj.type)
            hold on
            h1 = plot3(ParamTunObj.threshval,ParamTunObj.best_at_threshold(:,3),ParamTunObj.best_at_threshold(:,1),'o-','MarkerSize',5,'MarkerFaceColor',[0.3,0.3,0.3]);
            h2 = plot3(ParamTunObj.threshval_best,ParamTunObj.compval_best-0.1,ParamTunObj.perf_best,'bo','MarkerSize',10,'MarkerFaceColor',[0.7,1,1]);
            h3 = plot3(ParamTunObj.threshval_1se,ParamTunObj.compval_1se+0.1,ParamTunObj.perf_1se,'ro','MarkerSize',10,'MarkerFaceColor',[1,1,0.7]);
            legend([h1,h2,h3],{'best at threshold','Max Perf','1SE Perf'})
        end
    end
end

% checking input parameters to evalTuningParam to ensure smooth running
function [compvec,threshvec] = evalTuningParaminputchecking(cvmdl,type,X,Y,compvec,threshvec,subfold)
% 1. type checking
assert(isa(cvmdl,'TPLS_cv'),'First input should be a TPLS_cv model object');
assert(ismember(type,{'Pearson','Spearman','AUC','ACC','negMSE','negRMSE'}),'Unknown performance metric')
assert(all(isnumeric(X)) && all(isnumeric(Y)), 'Non numeric X Y. X and Y should be the same as the one used in TPLS_cv');
assert(all(isnumeric(compvec)),'compvec should be a numeric vector/scalar');
assert(all(isnumeric(threshvec)),'threshvec should be a numeric vector/scalar');
assert(all(isnumeric(subfold)),'subfold should be a numeric vector');

% 2. size checking
[n,v] = size(X); [nY,vY] = size(Y);
assert(v > 2 && n > 2 && n==nY && vY == 1,'Weird X Y dimensions. X and Y should be the same as the one used in TPLS_cv');
assert(length(compvec)==length(compvec(:)),'compvec should be a 1D numerical vector');
assert(length(threshvec)==length(threshvec(:)),'threshvec should be a 1D numerical vector');
assert(length(subfold)==length(subfold(:)) && length(subfold)==n,'subfold should be a 1D numerical vector of same length as Y');

% 3. nan checking
assert(~any(isnan(X(:))) && ~any(isnan(Y)),'NaN in X or Y.');
assert(~any(isnan(compvec)),'NaN in compvec.');
assert(~any(isnan(threshvec)),'NaN in threshvec.');
assert(~any(isnan(subfold)),'NaN in subfold.');

% 4. logic checking
assert(all(threshvec>=0) && all(threshvec<=1),'threshold values should be between 0 and 1, inclusive')
assert( max(compvec)<=cvmdl.NComp, 'compvec should only include values small than number of components used for TPLS_cv model');
compvec = sort(compvec(:)); threshvec = sort(threshvec(:));
end

function [perf_best,row_best,col_best,perf_1se,row_1se,col_1se,maxroute,maxrouteind] = findBestPerf(perfmat)
avgperfmat = nanmean(perfmat,3); % average performance
perf_best = max(avgperfmat(:)); % best point
[row_best,col_best] = find(avgperfmat==perf_best,1,'first'); % coordinates of best point
standardError = nanstd(squeeze(perfmat(row_best,col_best,:)))/sqrt(size(perfmat,3)); % finding the standard error of the best point
candidates = avgperfmat(:,1:col_best)>(perf_best-standardError); % finding points whose metric is higher than perf_max minus 1 SE
[row_1se,col_1se] = find(candidates,1,'first'); % coordinates of 1SE point
perf_1se = avgperfmat(row_1se,col_1se); % performance of 1SE point
[maxroute,maxrouteind] = max(avgperfmat); % finding the best map at each threshold point
end

function Perf = util_perfmetric(predmat,testY,type)
switch type
    case 'negMSE'
        Perf = -mean((predmat-testY).^2,1);
    case 'negRMSE'
        Perf = -sqrt(mean((predmat-testY).^2,1));
    case 'ACC'
        binarycheck(testY)
        Perf = mean(testY==1.*(predmat>0.5),1);
    case 'AUC'
        binarycheck(testY)
        n = size(testY,1); num_pos = sum(testY==1); num_neg = n - num_pos;
        Perf = 0.5 .* ones(1,size(predmat,2));
        if (num_pos>0 && num_pos < n)
            ranks = tiedrank(predmat); Perf = ( sum(ranks(testY==1,:),1) - num_pos * (num_pos+1)/2) / ( num_pos * num_neg);
        end
    case 'Pearson'
        Perf = corr(testY,predmat,'type',type);
    case 'Spearman'
        Perf = corr(testY,predmat,'type',type);
end
end

function binarycheck(Y)
uniqueY = unique(Y);
if length(uniqueY) > 2
    disp(uniqueY)
    error('non-binary Y detected')
end
if any(Y~=0 & Y~=1)
    disp(uniqueY)
    error('Y element is not 0 or 1')
end
end