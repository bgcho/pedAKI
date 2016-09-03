function boostedHIIOpts = boostedHII_setOpts(T)

defOpts = struct('T',T);         % The number of boosting rounds
defOpts.missingDataOpts = struct('method','abstain');
defOpts.stackingOpts = struct('use',0,...
                              'energyFrac',0.95);

boostedHIIOpts = defOpts;                          