% This splotByAge function groups feature matrix X, binary tag y, and weight w by the age_group

function [X_grouped, y_grouped, w_grouped] = splitByAge(X, y, w, predictors, age_group)
	if ~exist('w','var'); w = 1/length(y)*ones(length(y),1); end
	w_uq = unique(w);
	predictors = cellstr(predictors);
	idx_age = find(strcmp(predictors, 'age'));
	age_all = X(:, idx_age);
	X_grouped = {};
	y_grouped = {};
	w_grouped = {};
	% keyboard
	for ii = 1:size(age_group,1)
		if ii<size(age_group,1)
			age_mask = age_all>=age_group(ii,1) & age_all<age_group(ii,2);
		else
			age_mask = age_all>=age_group(ii,1) & age_all<=age_group(ii,2);
		end
		X_grouped{end+1} = X(age_mask, :);

		y_age = y(age_mask);
		y_age = y_age(:);
		y_grouped{end+1} = y_age;

		w_age = w(age_mask);
		w_age = w_age(:);
		for jj = 1:length(w_uq)
			nb_w_uq = sum(w_age==w_uq(jj));
			w_age(w_age==w_uq(jj)) = w_age(w_age==w_uq(jj))./nb_w_uq;
		end
		w_grouped{end+1} = w_age;
	end
end