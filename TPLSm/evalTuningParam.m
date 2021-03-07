classdef evalTuningParam
    properties
        type, threshval, compval, perfmat, perf_best, compval_best, threshval_best, perf_1se, compval_1se, threshval_1se, best_at_threshold;
    end
    methods
        function cvstats = evalTuningParam(cvmdl,type,X,Y,compvec,threshvec,subfold)
            % Evaluating cross-validation performance of a TPLS_cv model at compvec and threshvec
            %   'cvmdl'     : A TPLS_cv object
            %   'type'      : CV performance metric type. One of Pearson, Spearman, AUC, ACC, negMSE, negRMSE.
            %   'X'         : The same X as used in TPLS_cv.
            %   'Y'         : The same Y as used in TPLS_cv.
            %   'compvec'   : Vector of number of components to test in cross-validation.
            %   'threshvec' : Vector of threshold level [0 1] to test in cross-validation.
            %   'subfold'   : (Optional) vector of subdivision within testing fold to calculate performance. For example scan run division within subject.
            
            % input checking
            if nargin<7, subfold = ones(size(Y)); end
            assert(isa(cvmdl,'TPLS_cv'),'First input should be a TPLS_cv model object');
            assert(ismember(type,{'Pearson','Spearman','AUC','ACC','negMSE','negRMSE'}),'Unknown performance metric'); cvstats.type = type;
            TPLSinputchecker(X,'X','mat',[],[],1)
            TPLSinputchecker(Y,'Y','colvec',[],[],1)
            TPLSinputchecker(compvec,'compvec','vec',cvmdl.NComp,1,0,1); compvec = sort(compvec(:)); cvstats.compval = compvec;
            TPLSinputchecker(threshvec,'threshvec','vec',1,0); threshvec = sort(threshvec(:)); cvstats.threshval = threshvec;
            TPLSinputchecker(subfold,'subfold','vec')
            
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
            cvstats.perfmat = perfmat; avgperfmat = nanmean(perfmat,3); % mean performance
            cvstats.perf_best = max(avgperfmat(:)); % best mean performance
            [row_best,col_best] = find(avgperfmat==cvstats.perf_best,1,'first'); % coordinates of best point
            cvstats.compval_best = compvec(row_best); cvstats.threshval_best = threshvec(col_best); % component and threshold of best point
            SE = nanstd(perfmat(row_best,col_best,:),[],3)./sqrt(cvmdl.numfold); % standard error of best point
            [row_1se,col_1se] = find(avgperfmat(:,1:col_best)>(cvstats.perf_best-SE),1,'first'); % coordinates of 1SE point
            cvstats.perf_1se = avgperfmat(row_1se,col_1se); % performance of 1SE point
            cvstats.compval_1se = compvec(row_1se); cvstats.threshval_1se = threshvec(col_1se);
            [maxroute,maxrouteind] = max(avgperfmat);
            cvstats.best_at_threshold = [maxroute(:),threshvec,compvec(maxrouteind)];
        end
        function plot(cvstats)
            meansurf = nanmean(cvstats.perfmat,3);
            [X,Y] = meshgrid(cvstats.threshval,cvstats.compval);
            surf(X,Y,meansurf,'EdgeColor',[.5,.5,.5],'FaceAlpha',0.5,'FaceColor','interp')
            ylabel('Number of PLS components'); xlabel('Proportion of Voxels Left'); zlabel(cvstats.type)
            hold on
            h1 = plot3(cvstats.threshval,cvstats.best_at_threshold(:,3),cvstats.best_at_threshold(:,1),'o-','MarkerSize',5,'MarkerFaceColor',[0.3,0.3,0.3]);
            h2 = plot3(cvstats.threshval_best,cvstats.compval_best-0.1,cvstats.perf_best,'bo','MarkerSize',10,'MarkerFaceColor',[0.7,1,1]);
            h3 = plot3(cvstats.threshval_1se,cvstats.compval_1se+0.1,cvstats.perf_1se,'ro','MarkerSize',10,'MarkerFaceColor',[1,1,0.7]);
            legend([h1,h2,h3],{'best at threshold','Max Perf','1SE Perf'})
        end
    end
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