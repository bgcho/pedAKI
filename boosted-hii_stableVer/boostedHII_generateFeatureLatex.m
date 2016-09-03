function exprs = boostedHII_generateFeatureLatex(clf)
% u(x) = 2*I(x) - 1, where I(x) = 1 if x >= 0, 0 otherwise
% \sigma(x) = 2/(1+exp(-x)) - 1 (the sigmoid)

exprs = cell(numel(clf.weak_learners),1);
for j=1:numel(clf.weak_learners)
    bias = 0;
    
    expr = '';
    for k=1:numel(clf.weak_learners{j})
        c = clf.weak_learners{j}{k};
        if isfield(c,'bias')
            bias = bias + c.bias;
        end
        
        if k == 1
            prefixStr = '';
        else
            prefixStr = '&&';
        end
        
        switch c.type
            case 'constant'
                % Nothing left to do
            case 'stump'                
                expr = [expr,sprintf('%s%+.3fu(x_{%d} %+.3f) \\\\\n',prefixStr,c.alpha,j,-1*c.threshold)];
            case 'logistic'
                expr = [expr,sprintf('%s%+.3f\\sigma(%.3f(x_{%d} %+.3f)) \\\\\n',prefixStr,c.alpha,c.a,j,c.b/c.a)];
            otherwise
                error('Unknown weak learner type');
        end
    end
    if bias ~= 0
        expr = [expr,sprintf('&&%+.3f \\\\\n',bias)];
    end
    if ~numel(expr)
        expr = '0';
    end
    exprs{j} = [sprintf('\\begin{eqnarray*}\n f_{%d}(x_{%d}) &=& ',j,j),expr,sprintf('\\end{eqnarray*}')];
end

        