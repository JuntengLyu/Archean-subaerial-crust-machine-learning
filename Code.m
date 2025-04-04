% Data import
filename = 'train.xlsx'; % Archean igneous rocks for modeling
data_archean = readtable(filename);

filename3 = 'postArchean.xlsx'; % post-Archean igneous rocks for modeling
data_postarchean = readtable(filename3);

filename2 = 'predict.xlsx'; % Shale for prediction
predictdata = readtable(filename2);

filename5 = 'predictdia.xlsx'; % Glacial diamictites for prediction
predictdata_dia = readtable(filename5);

% Extract data
element_names = data_archean.Properties.VariableNames; % Obtain all indicator names
X_archean = data_archean{:, 2:end}; % All feature elements, except for the first column (MgO), Archean
y_archean = data_archean{:, 1}; % The first column (MgO), Archean

X_postarchean = data_postarchean{:, 2:end}; % All feature elements, except for the first column (MgO), post-Archean
y_postarchean = data_postarchean{:, 1}; % The first column (MgO), post-Archean

X_predict = predictdata{:, 2:end}; % All columns of shale except for the first column (Age)
Age = predictdata{:, 1}; % The first column (Age) of shale

X_predict_dia = predictdata_dia{:, 2:end}; % All columns of glacial diamictites except for the first column (Age)
Age_dia = predictdata_dia{:, 1};% The first column (Age) of glacial diamictites

% Randomly mix data: Generate samples
numMixSamples = 10000; % Number of mixed samples to generate

% Save the mixed sample data
mixedSamples_X_archean = zeros(numMixSamples, size(X_archean, 2)); 
mixedSamples_y_archean = zeros(numMixSamples, 1);

mixedSamples_X_postarchean = zeros(numMixSamples, size(X_postarchean, 2)); 
mixedSamples_y_postarchean = zeros(numMixSamples, 1);

% Mix Archean igneous rock data
for i = 1:numMixSamples
    mixNumPerSample = randi([2, 100]); % Each mixed sample is randomly composed of 2 to 100 original samples.
    selectedIdx = randperm(size(X_archean, 1), mixNumPerSample); % Randomly select sample indices
    mixRatios = rand(1, mixNumPerSample); % Randomly generate weights
    mixRatios = mixRatios / sum(mixRatios); % Normalization
    mixedSamples_X_archean(i, :) = sum(X_archean(selectedIdx, :) .* mixRatios', 1); % Mixed element concentrations
    mixedSamples_y_archean(i) = sum(y_archean(selectedIdx) .* mixRatios'); % Mixed MgO (y)
end
 
% Mix post-Archean igneous rock data
for i = 1:numMixSamples
    mixNumPerSample = randi([2, 100]); % Each mixed sample is randomly composed of 2 to 100 original samples.
    selectedIdx = randperm(size(X_postarchean, 1), mixNumPerSample); % Randomly select sample indices
    mixRatios = rand(1, mixNumPerSample); % Randomly generate weights
    mixRatios = mixRatios / sum(mixRatios); % Normalization
    mixedSamples_X_postarchean(i, :) = sum(X_postarchean(selectedIdx, :) .* mixRatios', 1); % Mixed element concentrations
    mixedSamples_y_postarchean(i) = sum(y_postarchean(selectedIdx) .* mixRatios'); % Mixed MgO (y)
end

% Data split (randomly): 70% for training, 30% for testing (using the mixed data)
cv_postarchean = cvpartition(size(mixedSamples_y_archean, 1), 'HoldOut', 0.3); % postArchean gets 70% data
idxTrain_postarchean = training(cv_postarchean);
idxTest_postarchean = test(cv_postarchean);

X_train_postarchean = mixedSamples_X_postarchean(idxTrain_postarchean, :);
y_train_postarchean = mixedSamples_y_postarchean(idxTrain_postarchean);
X_test_postarchean = mixedSamples_X_postarchean(idxTest_postarchean, :);
y_test_postarchean = mixedSamples_y_postarchean(idxTest_postarchean);

cv_archean = cvpartition(size(mixedSamples_y_postarchean, 1), 'HoldOut', 0.3); % Archean gets 70% data
idxTrain_archean = training(cv_archean);
idxTest_archean = test(cv_archean);

X_train_archean = mixedSamples_X_archean(idxTrain_archean, :);
y_train_archean = mixedSamples_y_archean(idxTrain_archean);
X_test_archean = mixedSamples_X_archean(idxTest_archean, :);
y_test_archean = mixedSamples_y_archean(idxTest_archean);

% Merge training data
X_train = [X_train_postarchean; X_train_archean];
y_train = [y_train_postarchean; y_train_archean];

% Merge testing data
X_test = [X_test_postarchean; X_test_archean];
y_test = [y_test_postarchean; y_test_archean];

X_predict = predictdata{:, 2:end};
Age = predictdata{:, 1};

X_predict_dia = predictdata_dia{:, 2:end};
Age_dia = predictdata_dia{:, 1};
%% Random Forest

% Input parameters
numFeatures = 8; % Specify the number of features
numTrees = 600; % Specify the number of trees
minLeafSize = 1; % Minimum split node size

% Print input parameters
fprintf('Number of features: %d\n', numFeatures);
fprintf('Number of trees: %d\n', numTrees);
fprintf('Minimum split node size: %d\n', minLeafSize);

% Train the random forest model
fprintf('Training the random forest model, please wait…\n');
rfModel = TreeBagger(numTrees, X_train, y_train, ...
    'Method', 'regression', ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on', ... % Enable feature importance calculation
    'MinLeafSize', minLeafSize);

% Calculate the correlation using OOB (Out-of-Bag) prediction results
y_oob_pred = oobPredict(rfModel);
oobCorr = corr(y_train, y_oob_pred);

fprintf('Model training is complete.\n');

% Training set
y_train_pred = predict(rfModel, X_train);
y_train_pred_postarchean = predict(rfModel, X_train_postarchean);
y_train_pred_archean = predict(rfModel, X_train_archean);

% Test set
y_test_pred = predict(rfModel, X_test);
y_test_pred_postarchean = predict(rfModel, X_test_postarchean);
y_test_pred_archean = predict(rfModel, X_test_archean);

% Calculate R2 and RMSE
R2_train = 1 - sum((y_train - y_train_pred).^2) / sum((y_train - mean(y_train)).^2);
RMSE_train = sqrt(mean((y_train - y_train_pred).^2));

R2_test = 1 - sum((y_test - y_test_pred).^2) / sum((y_test - mean(y_test)).^2);
RMSE_test = sqrt(mean((y_test - y_test_pred).^2));

fprintf('Training set performance:\n');
fprintf('  R2: %.4f\n', R2_train);
fprintf('  RMSE: %.4f\n', RMSE_train);

fprintf('Test set performance:\n');
fprintf('  R2: %.4f\n', R2_test);
fprintf('  RMSE: %.4f\n', RMSE_test);
%% Output important features
feature_importance = rfModel.OOBPermutedPredictorDeltaError;
[~, idx] = sort(feature_importance, 'descend');
fprintf('Feature importance ranking:\n');
for i = 1:length(idx)
    fprintf('%s: %.4f\n', element_names{idx(i)+1}, feature_importance(idx(i)));
end

% Plot a bar chart of feature importance
figure('Position', [584, 495, 600, 350]);
bar_handle = bar(feature_importance(idx));
bar_handle.FaceColor = [1 0.42156862745098 0.658823529411765];
set(gca, 'XTickLabel', element_names(idx+1), 'XTick', 1:length(idx));
xtickangle(45);
ylabel('Feature relative importance', 'FontSize', 14);
ax = gca;
ax.YLim = [0, 2.3];

avg_importance = mean(feature_importance(idx));
hold on;

avg_line = yline(avg_importance, '-', sprintf('Average = %.2f  ', avg_importance), 'LineWidth', 1);
avg_line.Color = [0.070588235294118 0.72156862745098 1];
hold off;
%% Plot: Training set
figure('Position', [584,495,520,420]);

errors1 = y_train_pred - y_train;
SE1 = std(errors1);
SE_upper1 = 2 * SE1;
SE_lower1 = -2 * SE1;

x_range1 = [min(y_train), max(y_train)];
fill([x_range1, fliplr(x_range1)], [x_range1 + SE_upper1, fliplr(x_range1 + SE_lower1)], ...
    [1, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold on;

plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', 'LineWidth', 2, 'Color', [1, 0, 0]);
hold on;

scatter(y_train_archean,y_train_pred_archean, 45,'MarkerEdgeAlpha',0.5,'MarkerEdgeColor','none',...
    'MarkerFaceAlpha',0.5,...
    'MarkerFaceColor',[1 0.32156862745098 0.658823529411765]);
hold on;

scatter(y_train_postarchean,y_train_pred_postarchean,45,'MarkerEdgeAlpha',0.5,'MarkerEdgeColor','none',...
    'MarkerFaceAlpha',0.5,...
    'MarkerFaceColor',[0.270588235294118 0.72156862745098 1]);

set(gca, 'YLim', [0 35], ...
         'XLim', [0 35]);
xlabel('Observed MgO (wt.%)', 'FontSize', 14,'FontWeight','bold');
ylabel('Predicted MgO (wt.%)', 'FontSize', 14,'FontWeight','bold');
hold off;
%% Plot: Test set
figure('Position', [584,495,520,420]);

errors2 = y_test_pred - y_test;
SE2 = std(errors2);
SE_upper2 = 2 * SE2;
SE_lower2 = -2 * SE2;

x_range2 = [min(y_test), max(y_test)];
fill([x_range2, fliplr(x_range2)], [x_range2 + SE_upper2, fliplr(x_range2 + SE_lower2)], ...
    [1, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold on;

plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', 'LineWidth', 2, 'Color', [1, 0, 0]);
hold on;

scatter(y_test_archean, y_test_pred_archean, 45, 'MarkerEdgeAlpha',0.5,'MarkerEdgeColor','none',...
    'MarkerFaceAlpha',0.5,...
    'MarkerFaceColor',[1 0.32156862745098 0.658823529411765]);
hold on;

scatter(y_test_postarchean, y_test_pred_postarchean, 45, 'MarkerEdgeAlpha',0.5,'MarkerEdgeColor','none',...
    'MarkerFaceAlpha',0.5,...); 
    'MarkerFaceColor',[0.270588235294118 0.72156862745098 1]);
set(gca, 'YLim', [0 25], ...
         'XLim', [0 25]);
xlabel('Observed MgO (wt.%)', 'FontSize', 14,'FontWeight','bold');
ylabel('Predicted MgO (wt.%)', 'FontSize', 14,'FontWeight','bold');
hold off;
%% Intraplate
filename_subduction1 = 'intraplate_data.xlsx';
data_in = readmatrix(filename_subduction1);
X_in = data_in(:, 2:end);
y_in = data_in(:, 1);

numMixSamples = 10000;

mixedSamples_X_in = zeros(numMixSamples, size(X_in, 2)); 
mixedSamples_y_in = zeros(numMixSamples, 1);

for i = 1:numMixSamples
    mixNumPerSample = randi([2, 100]);
    selectedIdx = randperm(size(data_in, 1), mixNumPerSample);
    mixRatios = rand(1, mixNumPerSample);
    mixRatios = mixRatios / sum(mixRatios);
    mixedSamples_X_in(i, :) = sum(X_in(selectedIdx, :) .* mixRatios', 1);
    mixedSamples_y_in(i) = sum(y_in(selectedIdx) .* mixRatios');
end

X_basalt_part1 = mixedSamples_X_in;
y_basalt_part1 = mixedSamples_y_in;

y_pred_basalt_part1 = predict(rfModel, X_basalt_part1);

R2_basalt_part1 = 1 - sum((y_basalt_part1 - y_pred_basalt_part1).^2) / sum((y_basalt_part1 - mean(y_basalt_part1)).^2);
RMSE_basalt_part1 = sqrt(mean((y_basalt_part1 - y_pred_basalt_part1).^2));

figure('Position', [584,495,520,420]);

x_range2 = [min(y_basalt_part1), max(y_basalt_part1)];
fill([x_range2, fliplr(x_range2)], [x_range2 + 2, fliplr(x_range2 - 2)], ...
    [1,0,0], 'FaceAlpha', 0.05, 'EdgeColor', 'none');

hold on;
fill([x_range2, fliplr(x_range2)], [x_range2 + 1, fliplr(x_range2 - 1)], ...
    [1,0,0], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold on;

plot(x_range2, x_range2, '--', 'LineWidth', 2, 'Color', [1, 0, 0]); % 红色虚线
hold on;

scatter(y_basalt_part1, y_pred_basalt_part1, 25, 'MarkerEdgeAlpha',0.2,'MarkerEdgeColor','none',...
    'MarkerFaceAlpha',0.2,...
    'MarkerFaceColor',[1 0.32156862745098 0.658823529411765]);
hold on;

set(gca, 'YLim', [0 12]);
set(gca, 'XLim', [0 12]);
xlabel('Observed MgO (wt.%)', 'FontSize', 14,'FontWeight','bold');
ylabel('Predicted MgO (wt.%)', 'FontSize', 14,'FontWeight','bold');
title(sprintf('intraplate\nR^2: %.4f, RMSE: %.4f', R2_basalt_part1, RMSE_basalt_part1));
hold off;
%% convergent

filename_subduction1 = 'Chen Kang.xlsx'; 
data_arc = readmatrix(filename_subduction1);
X_arc = data_arc(:, 2:end);
y_arc = data_arc(:, 1);

numMixSamples = 10000;

mixedSamples_X_arc = zeros(numMixSamples, size(X_arc, 2)); 
mixedSamples_y_arc = zeros(numMixSamples, 1);

for i = 1:numMixSamples
    mixNumPerSample = randi([2, 100]);
    selectedIdx = randperm(size(data_arc, 1), mixNumPerSample);
    mixRatios = rand(1, mixNumPerSample);
    mixRatios = mixRatios / sum(mixRatios);
    mixedSamples_X_arc(i, :) = sum(X_arc(selectedIdx, :) .* mixRatios', 1);
    mixedSamples_y_arc(i) = sum(y_arc(selectedIdx) .* mixRatios');
end

X_diorite_intra = mixedSamples_X_arc;
y_diorite_intra = mixedSamples_y_arc;
y_pred_diorite_intra = predict(rfModel, X_diorite_intra);

R2_diorite_intra = 1 - sum((y_diorite_intra - y_pred_diorite_intra).^2) / sum((y_diorite_intra - mean(y_diorite_intra)).^2);
RMSE_diorite_intra = sqrt(mean((y_diorite_intra - y_pred_diorite_intra).^2));

figure('Position', [584,495,520,420]);

x_range2 = [min(y_basalt_part1), max(y_basalt_part1)];
fill([x_range2, fliplr(x_range2)], [x_range2 + 2, fliplr(x_range2 - 2)], ...
    [0,0.65,1], 'FaceAlpha', 0.05, 'EdgeColor', 'none');
hold on;

fill([x_range2, fliplr(x_range2)], [x_range2 + 1, fliplr(x_range2 - 1)], ...
    [0,0.65,1], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold on;

plot(x_range2, x_range2, '--', 'LineWidth', 2, 'Color', [0, 0.47, 1]);
hold on;

scatter(y_diorite_intra, y_pred_diorite_intra, 25, 'MarkerEdgeAlpha',0.2,'MarkerEdgeColor','none',...
    'MarkerFaceAlpha',0.2,...
    'MarkerFaceColor',[0.270588235294118 0.72156862745098 1]);
hold on;

set(gca, 'YLim', [0 10]);
set(gca, 'XLim', [0 10]);
xlabel('Observed MgO (wt.%)', 'FontSize', 14,'FontWeight','bold');
ylabel('Predicted MgO (wt.%)', 'FontSize', 14,'FontWeight','bold');
title(sprintf('convergent margin\nR^2: %.4f, RMSE: %.4f', R2_diorite_intra, RMSE_diorite_intra));
hold off;
%% MgO content in the continental crust over time
MgO_cal = predict(rfModel, X_predict);
MgO_cal_dia = predict(rfModel, X_predict_dia);

figure('Position', [584,495,560,420]);

scatter(Age, MgO_cal, 50, ...
    'MarkerFaceColor', [1 0.7058824 0.8901961], ...
    'MarkerEdgeColor', [1 0.32156862745098 0.758823529411765], ...
    'LineWidth', 1.5);
hold on;

scatter(Age_dia, MgO_cal_dia, 50, ...
    'MarkerFaceColor', [0.63,0.85,1.00], ...
    'MarkerEdgeColor', [0.1,0.65,1.00], ...
    'LineWidth', 1.5);

set(gca, 'FontName', 'Arial',...
         'FontSize', 12,...
         'YLim', [0 20]);
xlabel('Depositional age (Ma)', 'FontSize', 14,'FontWeight','bold');
ylabel('Predicted MgO (wt.%) in the UCC', 'FontSize', 14,'FontWeight','bold');
box on;
hold off;
%% MgO content in the continental crust by Age-bin
MgO_cal_all = [MgO_cal; MgO_cal_dia];
Age_all = [Age; Age_dia];

bin_edges = 3500:-500:0;
bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;

avg_MgO = zeros(1, length(bin_centers));
se_MgO = zeros(1, length(bin_centers));

for i = 1:length(bin_edges)-1
    in_bin = (Age_all > bin_edges(i+1)) & (Age_all <= bin_edges(i));
    bin_data = MgO_cal_all(in_bin);
    avg_MgO(i) = mean(bin_data, 'omitnan');
    se_MgO(i) = std(bin_data, 'omitnan') / sqrt(sum(~isnan(bin_data)));
end

figure('Position', [584,495,350,200]); % [x, y, width, height]

line_length = 70;

for i = 1:length(bin_centers)
    line([bin_centers(i), bin_centers(i)], ...
         [avg_MgO(i) - 2 * se_MgO(i), avg_MgO(i) + 2 * se_MgO(i)], ...
         'Color', [0.02,0.42,0.09], 'LineWidth', 1.5); 
    line([bin_centers(i) - line_length / 2, bin_centers(i) + line_length / 2], ...
         [avg_MgO(i) + 2 * se_MgO(i), avg_MgO(i) + 2 * se_MgO(i)], ...
         'Color', [0.02,0.42,0.09], 'LineWidth', 1.5); 
    line([bin_centers(i) - line_length / 2, bin_centers(i) + line_length / 2], ...
         [avg_MgO(i) - 2 * se_MgO(i), avg_MgO(i) - 2 * se_MgO(i)], ...
         'Color', [0.02,0.42,0.09], 'LineWidth', 1.5); 
end
hold on;

scatter(bin_centers, avg_MgO, 90, ...
    'MarkerFaceColor', [0.64 1.00 0.65], ...
    'MarkerEdgeColor', [0.07 0.66 0.00], ...
    'LineWidth', 1.5);

set(gca, 'FontName', 'Arial',...
         'FontSize', 12,...
         'YLim', [1 11], ...
         'XLim', [0 3500]);
hold off;
box on;
ylabel('Mean predicted MgO (wt.%)','FontWeight','bold');

%% Calculate the proportion of different types of rocks
MgO_cal_all = [MgO_cal; MgO_cal_dia];
Age_all = [Age; Age_dia];

bin_edges_1 = 3500:-200:2500;
bin_centers_1 = (bin_edges_1(1:end-1) + bin_edges_1(2:end)) / 2;
bin_edges_2 = 2500:-200:0;
bin_centers_2 = (bin_edges_2(1:end-1) + bin_edges_2(2:end)) / 2;

granite_archean = 1.1;
granite_postarchean = 1.6;

basalt_archean = 7.6;
basalt_postarchean = 6.9;

komatiite_archean = 26;
komatiite = [0 0.025 0.05 0.075 0.1];

figure;
hold on;

for i = 1:length(bin_edges_1)-1
    in_bin = (Age_all > bin_edges_1(i+1)) & (Age_all <= bin_edges_1(i));
    bin_data = MgO_cal_all(in_bin);
    avg_MgO_1 = mean(bin_data, 'omitnan');

    for k = 1:length(komatiite)
        granite_1 = (basalt_archean - basalt_archean * komatiite(k) - avg_MgO_1 + komatiite_archean * komatiite(k)) / (basalt_archean - granite_archean);
        basalt_1 = (granite_archean - avg_MgO_1 + komatiite_archean * komatiite(k) - granite_archean * komatiite(k)) / (granite_archean - basalt_archean);

        scatter(bin_centers_1(i), granite_1, 50, [1-komatiite(k), komatiite(k), 0], 'filled');
        scatter(bin_centers_1(i), basalt_1, 50, [0, 0, 1-komatiite(k)], 'filled');

        text(bin_centers_1(i), granite_1, sprintf('%.3f', komatiite(k)), 'Color', [1-komatiite(k), komatiite(k), 0], 'FontSize', 8, 'HorizontalAlignment', 'center');
        text(bin_centers_1(i), basalt_1, sprintf('%.3f', komatiite(k)), 'Color', [0, 0, 1-komatiite(k)], 'FontSize', 8, 'HorizontalAlignment', 'center');
    end
end

for i = 1:length(bin_edges_2)-1
    in_bin = (Age_all > bin_edges_2(i+1)) & (Age_all <= bin_edges_2(i));
    bin_data = MgO_cal_all(in_bin);
    avg_MgO_2 = mean(bin_data, 'omitnan');

    for k = 1:length(komatiite)
        granite_2 = (basalt_postarchean - basalt_postarchean * komatiite(k) - avg_MgO_2 + komatiite_archean * komatiite(k)) / (basalt_postarchean - granite_postarchean);
        basalt_2 = (granite_postarchean - avg_MgO_2 + komatiite_archean * komatiite(k) - granite_postarchean * komatiite(k)) / (granite_postarchean - basalt_postarchean);

        scatter(bin_centers_2(i), granite_2, 50, [1-komatiite(k), komatiite(k), 0], 'filled');
        scatter(bin_centers_2(i), basalt_2, 50, [0, 0, 1-komatiite(k)], 'filled');

        text(bin_centers_2(i), granite_2, sprintf('%.3f', komatiite(k)), 'Color', [1-komatiite(k), komatiite(k), 0], 'FontSize', 8, 'HorizontalAlignment', 'center');
        text(bin_centers_2(i), basalt_2, sprintf('%.3f', komatiite(k)), 'Color', [0, 0, 1-komatiite(k)], 'FontSize', 8, 'HorizontalAlignment', 'center');
    end
end
hold off;

xlim([2300,3300]);
ylim([0,1]);
box on;
xlabel('Age (Ma)','FontSize',14,'FontWeight','bold');
ylabel('Percentage','FontSize',14,'FontWeight','bold');