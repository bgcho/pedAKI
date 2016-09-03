function boostedHII_generateStackingLatex(clf,fileName,mode)

if ~exist('mode','var'); mode = 'csv'; end;

p = numel(clf.weak_learners) - 1;

if strcmp(mode,'latex') 
    S_expr = sprintf('S = \\left[\\begin{array}{%s}\n',repmat('c',1,p+1));
elseif strcmp(mode,'csv')
    S_expr = '';
elseif strcmp(mode,'mat')
    S = zeros(p+1,p+1);
else
    error('Unknown mode');
end

for j = [numel(clf.weak_learners),1:numel(clf.weak_learners)-1]
    % Do the bias term first

    % Combine the first and last stacking coefficients -- these are 
    mf_coeffs = [clf.stacking.mf_coeffs{j}(1) + clf.stacking.mf_coeffs{j}(end);clf.stacking.mf_coeffs{j}(2:end-1)'];
    for k=1:numel(mf_coeffs)
        if strcmp(mode,'latex')
            if k > 1            
                S_expr = [S_expr,' & ',sprintf('%.3f',mf_coeffs(k))];
            else
                S_expr = [S_expr,sprintf('%.3f',mf_coeffs(k))];
            end
        elseif strcmp(mode,'csv')
            if k > 1
                S_expr = [S_expr,',',sprintf('%f',mf_coeffs(k))];
            else
                S_expr = [S_expr,sprintf('%f',mf_coeffs(k))];
            end
        elseif strcmp(mode,'mat')
            S(j,k) = mf_coeffs(k);
        else
            error('Unknown mode');
        end
    end
    if strcmp(mode,'latex')
        S_expr = [S_expr,sprintf(' \\\\\n')];
    elseif strcmp(mode,'csv')
        S_expr = [S_expr,sprintf('\n')];
    elseif strcmp(mode,'mat')
        % Don't do anything here
    else
        error('Unknown mode');
    end
end
if strcmp(mode,'latex')
    S_expr = [S_expr,sprintf('\\end{array}\\right]')];
elseif strcmp(mode,'csv')
    % Don't do anything
elseif strcmp(mode,'mat')
     % Don't do anything
else
    error('Unknown mode');
end

if strcmp(mode,'latex') || strcmp(mode,'csv')
    fp = fopen(fileName,'w');
    fprintf(fp,'%s',S_expr);
    fclose(fp);
elseif strcmp(mode,'mat')
    S = S([end,1:end-1],:);
    save(fileName,'S');
else
    error('Unknown mode');
end

