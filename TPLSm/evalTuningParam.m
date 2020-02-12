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