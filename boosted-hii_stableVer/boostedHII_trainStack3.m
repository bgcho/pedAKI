function stacking = boostedHII_trainStack3(X,y,weak_learners,stackingOpts)
    
if ~stackingOpts.use
    stacking = [];
    return;
end

[n,p] = size(X);
if p == numel(weak_learners) - 1
    X = [X,ones(n,1)];
    p = p + 1;
end

uy = sort(unique(y),'ascend');
if uy(1) == 0 && uy(2) == 1
    y = 2*y - 1;
elseif uy(1) == -1 && uy(2) == 1
    % We are fine
else
    error('Unknown type for y');
end

feats_picked = zeros(p,1);
for j=1:numel(weak_learners)
    if numel(weak_learners{j})
        feats_picked(j) = 1;
    end
end
feats_picked = find(feats_picked);
feats_notpicked = setdiff(1:p,feats_picked);

M = double(isnan(X));

clos = zeros(n,p);
for j=1:p
    clos(:,j) = boostedHII_evaluateWeakLearner(X,weak_learners,j);
end
clo = sum(clos,2);

w = exp(-y.*clo); w = w/sum(w);

mf_coeffs = cell(p,1);
for j=1:p
    mf_coeffs{j} = [1,zeros(1,p)];
end

stacking_order = [];
C = zeros(n,size(clos,2)*(1+size(M,2)));
eInd = 0;
for j=1:p
    sInd = eInd + 1;
    eInd = sInd + size(M,2);
    C(:,sInd:eInd) = repmat(clos(:,j),1,size(M,2)+1).*[ones(n,1),M];
end
C(:,mean(C~=0)<0.2) = 0;
C_unnorm = C;
cmags = sqrt(sum(C.^2));
locs = find(cmags > 1e-6);
C(:,locs) = C(:,locs)./repmat(cmags(locs),n,1);

for t=1:200
    fprintf(1,'%d\n',t);pause(1e-5);
    
    % Reset the bias
    b = resetBias(y,w);
    w = w.*exp(-y*b); w = w/sum(w);
    mf_coeffs{end}(1) = (mf_coeffs{end}(1)*clos(1,end) + b)/clos(1,end);
    clo = clo + b;
    
    wy = w.*y;
    
    if 1
        [~,ind] = max(abs(wy'*C));
        best_c = C_unnorm(:,ind);
        % Convert ind to j and k
        [k,j] = ind2sub([p+1,p],ind);
        k = k - 1;
    else
        j = 0;
        k = 0;
        best_c = [];
        best_innerprod = 0;
        for j2=1:p
            for k2=0:p
                if k2 == 0
                    c = clos(:,j2);
                else
                    c = clos(:,j2).*M(:,k2);
                end
                innerprod = abs(sum(wy.*(c./sqrt(sum(c.^2)))));
                if innerprod > best_innerprod
                    best_c = c;
                    j = j2;
                    k = k2;
                    best_innerprod = innerprod;
                end
            end
        end
    end
    
    alpha = alpha_linesearch(w,y,best_c);
    clo_curr = alpha*best_c;
    w = w.*exp(-y.*clo_curr); w = w/sum(w);

    
    stacking_order(t).j = j;
    stacking_order(t).k = k;
    stacking_order(t).alpha = alpha;
    
    if k == 0
        mf_coeffs{j}(1) = mf_coeffs{j}(1) + alpha;
    else
        mf_coeffs{j}(1) = mf_coeffs{j}(1) + alpha;
        mf_coeffs{j}(k+1) = mf_coeffs{j}(k+1) - alpha;
    end
    
    clo = clo + clo_curr;
    printmsg('Objective function at stacking round #%d:  %.2f\n',t,objective_function(y,clo));
    printmsg('Training error at stacking round #%d:  %.2f\n',t,training_error(y,clo));
end
stacking.stacking_order = stacking_order;
stacking.mf_coeffs = mf_coeffs;
end

function alpha = alpha_linesearch(w,y,z)
    % Now compute alpha
    % Need to use binary search for this
    % We are looking for a root of the derivative
    % d/d_alpha = -sum(w(i)*y(i)*z(i)*exp(-y(i)*z(i)*alpha))
    u = y.*z;
    alpha_lb = -inf;
    alpha_ub = inf; alpha = 0;
    while 1
        Zd = -1*sum(w.*u.*exp(-alpha*u));
        if abs(Zd) < 1e-8
            break;
        end
        if Zd > 0
            alpha_ub = alpha;
            if isinf(alpha_lb)
                alpha = alpha - 1;
            else
                alpha = 0.5*(alpha_lb + alpha_ub);
            end
        else
            alpha_lb = alpha;
            if isinf(alpha_ub)
                alpha = alpha + 1;
            else
                alpha = 0.5*(alpha_lb + alpha_ub);
            end
        end
    end
end


    function printmsg(msg,varargin)
        %if verbosity >= 1
            fprintf(1,msg,varargin{:});
        %end
    end

    function obj_func = objective_function(y,clo)
        obj_func = sum(exp(-y.*clo));
    end

    function train_err = training_error(y,clo)
        train_err = sum(sign(clo)~=y)/numel(y);
    end


function alpha = resetBias(y,w,b)
% Picks the appropriate weighting on a constant output
% so that after updating weights, we have sum(w(y==1)) == sum(w(y==-1))
% b is the threshold
    if ~exist('b','var')
        b = 0;
    end
    
    frac_neg1 = sum(w(y==-1));
    frac_pos1 = sum(w(y==1));
        
    if b < 1
        alpha = 0.5*log(frac_pos1/frac_neg1);
    elseif b > 1
        alpha = 0.5*log(frac_neg1/frac_pos1);
    else
        error('b cannot be 1');
    end
end

