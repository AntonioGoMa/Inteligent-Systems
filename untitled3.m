csvFile = fullfile('labels.csv');

% Load the dataset
trainFileName = 'labels.csv'; 
trainLabel = 'labels.csv'; 

trainingData = loadDatasetFromCSV(trainFileName);
validationLabels = loadDatasetFromCSV(trainLabel);


% Pre-process the dataset
inputSize = [224, 224, 3]; 
for i = 1:size(trainingData, 1)
    trainingData.image{i} = preprocessData(trainingData.image{i}, inputSize);
end
for i = 1:size(validationLabels, 1)
    validationLabels.image{i} = preprocessData(validationLabels.image{i}, inputSize);
end

% Load and customize the pre-trained network
net = resnet50; 
lgraph = modifyPretrainedNetwork(net);

% training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validationLabels, ...
    'ValidationFrequency', 30, ...
    'Verbose', true);

% Train the model
[trainedNet, trainInfo] = trainNetwork(trainingData, lgraph, options);

sampleNumber = size(validationLabels, 1);
IoU_scores = zeros(sampleNumber, 1);

for i = 1:sampleNumber
    % Get the ground truth bounding box
    groundTruthBbox = validationLabels.bbox{i};

    % Predict the bounding box using the trained network
    inputImage = validationLabels.image{i};
    predictedBbox = predict(trainedNet, inputImage);

    % Calculate the IoU score
    IoU_scores(i) = bboxIoU(groundTruthBbox, predictedBbox);
end

% Compute the average IoU score
meanIoU = mean(IoU_scores);

fprintf('Mean Intersection over Union (IoU) on the validation set: %.4f\n', meanIoU);


% Initialize empty arrays for predicted labels and ground truth labels
predictL = [];
grountT = [];

% Iterate over validation data and make predictions
for i = 1:sampleNumber
    % Get the input image
    inputImage = validationLabels.image{i};
    
    % Preprocess the input image
    preprocessedImage = preprocessData(inputImage, inputSize);
    
    % Predict the label using the trained network
    scores = predict(trainedNet, preprocessedImage);
    binaryLabel = scores > 0.5;
    predictL = [predictL; binaryLabel];
    groundTruthLabel = validationLabels.bbox{i}; 
    % Append the ground truth label to groundTruthLabels
    grountT = [grountT; groundTruthLabel];
end

% Calculate TP, FP, and FN
trueP = sum(predictL & grountT);
falseP = sum(predictL & ~grountT);
FN = sum(~predictL & grountT);
TN = sum(~predictL & ~grountT);

% Calculate precision
precision = trueP ./ (trueP + falseP);

% Calculate recall
recall = trueP ./ (trueP + FN);

% Calculate F1 score
f1Score = 2 * (precision .* recall) ./ (precision + recall);

% Print precision and F1 score
fprintf('Precision: %.4f\n', precision*100);
fprintf('F1 Score: %.4f\n', f1Score *100);

% Calculate accuracy
accuracy = (trueP + TN) ./ (trueP + TN + falseP + FN);

% Print accuracy
fprintf('Accuracy: %.4f\n', accuracy*100);

figure(6)
confusionchart(grountT, predictL, 'Title', 'Performance');
fprintf('Accuracy: %.2f%%\n', 100 * accuracy);
fprintf('Precision: %.4f\n', precision*100);
fprintf('F1 Score: %.4f\n', f1Score *100);
printf(meanIoU);

%preprocess data
function dataset = loadDatasetFromCSV(csvFile)
    T = readtable(csvFile);
    numEntries = size(T, 1);
    dataset = table(cell(numEntries, 1), cell(numEntries, 1), 'VariableNames', {'image', 'bbox'});
    for i = 1:numEntries
        dataset.image{i} = imread(T.filepath{i}); 
        dataset.bbox{i} = reshape([T.xmin(i), T.ymin(i), T.xmax(i), T.ymax(i)], [1, 1, 4]);
    end
end


function dataOut = preprocessData(dataIn, inputSize)
    dataOut = imresize(dataIn, inputSize(1:2));
end

function lgraph = modifyPretrainedNetwork(net)
    % Load the pre-trained network
    lgraph = layerGraph(net);

    % Remove the last layers
    lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

    % Add new layers for bounding box regression
    numOutputs = 4; % xmin, ymin, width, and height
    newOutputLayer = fullyConnectedLayer(numOutputs, 'Name', 'bbox_output');
    newRegLayer = regressionLayer('Name', 'RegressionLayer_bboxes');

    % Connect the new layers to the network
    lgraph = addLayers(lgraph, [newOutputLayer, newRegLayer]);
    lgraph = connectLayers(lgraph, 'avg_pool', 'bbox_output');
end




function iou = bboxIoU(bbox1, bbox2)
    x_left = max(bbox1(1), bbox2(1));
    y_top = max(bbox1(2), bbox2(2));
    x_right = min(bbox1(3), bbox2(3));
    y_bottom = min(bbox1(4), bbox2(4));

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1);
    bbox1_area = (bbox1(3) - bbox1(1) + 1) * (bbox1(4) - bbox1(2) + 1);
    bbox2_area = (bbox2(3) - bbox2(1) + 1) * (bbox2(4) - bbox2(2) + 1);

    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area);
end




