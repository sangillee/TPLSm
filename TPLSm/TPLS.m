classdef TPLS %   Thresholded Partial Least Squares. -Arthur Lee
    properties
        NComp, W, MtrainX, MtrainY, scoreCorr, pctVar, betamap, threshmap
    end
    methods
        function TPLSmdl = TPLS(X,Y,NComp,W,nmc) % NComp : maximum number of partial least squares components you want. W: n-by-1 observation weight vector.
            if nargin<3, NComp = 50; end % default value
            if nargin<4, W = ones(size(Y)); end % if weight is not provided, every observation has equal weight
            if nargin<5, nmc = 0; end % this is a special switch to skip mean centering
            assert( all(W>0) ,'weights should be positive'); W = W./sum(W); % normalize weight sum to 1
            
            % data dimensions and data type specification
            dt = superiorfloat(X,Y); [n,v] = size(X); if ~strcmp(superiorfloat(W),dt); W = cast(W,dt); end
            
            % Mean-Center variables as needed by SIMPLS algorithm
            TPLSmdl.NComp = NComp; TPLSmdl.W = W;
            TPLSmdl.MtrainX = W'*X; TPLSmdl.MtrainY = W'*Y; % calculating weighted means of X and Y
            if nmc == 0 % if no switch is given to skip mean centering
                X = bsxfun(@minus, X, TPLSmdl.MtrainX); Y = Y - TPLSmdl.MtrainY; % subtract means
            else
                disp('mean centering disabled')
                if mean(abs(TPLSmdl.MtrainX)) > 1e-04
                    warning('X does not seem to be mean-centered. Results may not be valid')
                end
            end
            
            % allocate memories for output variables, interim variables, and calculate often used variables
            TPLSmdl.scoreCorr = nan(NComp,1,dt); TPLSmdl.betamap = nan(v,NComp,dt); TPLSmdl.threshmap = [0.5 .* ones(v,1,dt), nan(v,NComp-1,dt)];
            B = nan(NComp,1,dt); P2 = nan(n,NComp,dt); C = nan(v,NComp,dt); sumC2 = zeros(v,1,dt); r = Y; V = nan(v,NComp);
            WYT = (W.*Y)'; WTY2 = W'*Y.^2; WT = W'; W2 = W.^2; % often-used variables
            
            % perform Arthur-modified SIMPLS algorithm
            Cov = (WYT*X)'; % weighted covariance
            for i = 1:NComp
                disp(['Calculating Comp #',num2str(i)])
                P = X*Cov; % this is the component, before normalization
                norm_P = sqrt(WT*P.^2); % weighted standard deviation of component as normalization constant
                P = P ./ norm_P; B(i) = (norm(Cov)^2)/norm_P; C(:,i) = Cov ./ norm_P; % normalize component, beta, and back-projection coefficient
                
                % Update the orthonormal basis with modified Gram Schmidt
                vi = ((W.*P)'*X)'; % weighted covariance between X and current component
                vi = vi - V(:,1:i-1)*(V(:,1:i-1)'*vi); % orthogonalize with regards to previous components
                vi = vi ./ norm(vi); V(:,i) = vi; % normalize and add to orthonormal basis matrix
                Cov = Cov - vi*(vi'*Cov); Cov = Cov - V(:,1:i)*(V(:,1:i)'*Cov); % Deflate Covariance using the orthonormal basis matrix
                
                % back-projection
                TPLSmdl.betamap(:,i) = C(:,1:i)*B(1:i); % back-projection of coefficients
                sumC2 = sumC2+C(:,i).^2; P2(:,i) = P.^2; r = r - P*B(i); % some variables that will facilitate computation later
                if i > 1 % no need to calculate threshold for first component
                    se = sqrt(P2(:,1:i)'*(W2.*(r.^2))); %Huber-White Sandwich estimator (assume no small T bias)
                    TPLSmdl.threshmap(:,i) = abs((C(:,1:i)*(B(1:i)./se))./sqrt(sumC2)); % absolute value of back-projected z-statistics
                end
            end
            TPLSmdl.threshmap(:,2:end) = (v-tiedrank(TPLSmdl.threshmap(:,2:end)))./v; % convert into thresholds between 0 and 1
            TPLSmdl.pctVar = B.^2 ./ WTY2; % Compute the percent of variance of Y each component explains
            TPLSmdl.scoreCorr = sqrt(TPLSmdl.pctVar); % weighted correlation between Y and current component
        end
        
        function [betamap,bias] = makePredictor(TPLSmdl,compval,threshval)
            % extract betamap from a TPLS model at a given number of components and at given threshold value
            assert( length(threshval)==1, 'only one threshold value should be used');
            betamap = TPLSmdl.betamap(:,compval);
            if threshval < 1
                betamap = betamap.*(TPLSmdl.threshmap(:,compval)<=threshval); % finalized predictor map at select components
            end
            bias = TPLSmdl.MtrainY-TPLSmdl.MtrainX*betamap; % post-fitting of bias
        end
        
        function score = predict(TPLSmdl,compval,threshval,testX)
            assert( length(threshval)==1, 'only one threshold value should be used');
            if threshval == 0
                score = repmat(TPLSmdl.MtrainY,size(testX,1),length(compval));
            else
                [threshbetamap,bias] = makePredictor(TPLSmdl,compval,threshval);
                score = bsxfun(@plus,bias,testX*threshbetamap);
            end
        end
    end
end