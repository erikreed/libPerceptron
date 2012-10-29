%% Erik Reed (erikreed@cmu.edu)

training = importdata('d2-train.csv');
Y = training(:,1);
X = training(:,2:end);

testing = importdata('d2-test.csv');
Yt = testing(:,1);
Xt = testing(:,2:end);

NUM_TRIALS = 500;
EPSILON = .001;
STEP_SIZE = .0001;


% logistic regression
X = [ones(size(X, 1), 1) X];
Xt = [ones(size(Xt, 1), 1) Xt];

% u = 1/(1+exp(-theta'*x_i))
% P(y|x) = u(x)^y(1-u(x))^(1-y)

% training
maxLambda = 19;
lambdaErr = zeros(1,maxLambda);
for lambda=1:maxLambda
    classified = zeros(length(Y),1);
    %
        theta = zeros(1,305);
        delta = inf;
        steps = 0;
        while delta > EPSILON
            steps = steps + 1;
            theta_old = theta;
            for j=1:size(X,1)
                theta_diff = STEP_SIZE * ...
                    (Y(j) - 1/(1+exp(-theta*X(j,:)')))*X(j,:);
                theta = theta + theta_diff - abs(theta_diff) * lambda / size(X,1);
            end

            delta = sqrt(sum((theta_old - theta).^2));
        end
        for k=1:50
        classified(k) = round(1/(1+exp(-theta*X(k,:)')));

    end
    
    trainingError = sum(classified ~= Y) / length(Y);
    lambdaErr(lambda) = trainingError;
end

% testing
classified = zeros(size(Xt, 1), 1);
for i=1:size(Xt, 1)
    classified(i) = round(1/(1+exp(-theta*Xt(i,:)')));
end
testError = sum(classified ~= Yt) / length(Yt);

