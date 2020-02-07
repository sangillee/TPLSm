classdef evalTuningParam
    properties
        type, threshval, compval, perfmat, maxperf, compval_max, threshval_max, compval_1se, threshval_1se;
    end
    methods
        function ParamTunObj = evalTuningParam(TPLScvmdl,type,X,Y,compvec,threshvec)
            
            ParamTunObj.type = type; % specify tuning performance type
            ParamTunObj.threshval = threshvec;
            ParamTunObj.compval = compvec;
            ParamTunObj.perfmat = nan(length(compvec),length(threshvec),TPLScvmdl.numfold);
            
            for i = 1:TPLScvmdl.numfold
                disp(['Fold #',num2str(i)])
                test = TPLScvmdl.testfold == i;
                for j = 1:length(threshvec)
                    predmat =  predict(TPLScvmdl.cvMdls{i},compvec,threshvec(j),X(test,:));
                    ParamTunObj.perfmat(:,j,i) = util_perfmetric(predmat,Y(test),type);
                end
            end
            avgperfmat = mean(ParamTunObj.perfmat,3);
            
            if ismember(type,{'MSE','RMSE','MAD'}) % for these metrics, lower value is better
                ParamTunObj.maxperf = min(avgperfmat(:));
            else
                ParamTunObj.maxperf = max(avgperfmat(:));
            end
            
            [row,col] = find(avgperfmat==ParamTunObj.maxperf);
            ParamTunObj.compval_max = compval(row);
            ParamTunObj.threshval_max = threshval(col);
            
            
        end
        function plot(ParamTunObj)
            figure
            [X,Y] = meshgrid(ParamTunObj.threshval,ParamTunObj.compval);
            surf(X,Y,avgperfmat,'EdgeColor',[.5,.5,.5],'FaceAlpha',0.5,'FaceColor','interp')
            ylabel('Number of PLS components'); xlabel('Proportion of Voxels Left'); zlabel(ParamTunObj.type)
            set(gca, 'XScale', 'log')
        end
    end
end