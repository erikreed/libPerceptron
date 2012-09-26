dat = importdata('data.csv');
classes_full = dat.textdata;
NUM_TRIALS = 100;
MAX_TRAINING = 200;
EPSILON = .001;
STEP_SIZE = .0001;

error_vs_training = zeros(1,MAX_TRAINING);
error_vs_trainingLR = zeros(1,MAX_TRAINING);

parfor m=2:MAX_TRAINING
    error_avg = zeros(1,NUM_TRIALS);
    error_avgLR = zeros(1,NUM_TRIALS);
    for trial=1:NUM_TRIALS
        rand_permutation = randperm(size(dat.data,1));
        training = dat.data(rand_permutation,:);
        classes = classes_full(rand_permutation, :);
        classesTesting = classes(length(classes)*2/3:end,:);
        classes = classes(1:length(classes)*2/3,:);
        testing = training(length(training)*2/3:end,:);
        training = training(1:length(training)*2/3,:);

        % prune dataset
        classes = classes(1:m,:);
        training = training(1:m,:);

        % -- Naive Bayes -- 

        theta = zeros(16,3,2);
        priors = zeros(1,2);

        % training
        for k='A':'B'
            priors(k - 'A' + 1) = sum(strcmp(classes,k)) / length(classes);
            % for each value of each feature
            for i=1:16
               for j=1:3
                  theta(i,j,k - 'A' + 1) =  max(sum(training(strcmp(classes, k), i) == j),1)/length(classes);
               end
            end
        end

        % testing
        classified = zeros(length(classesTesting), 1);
        for p=1:length(classified)
            best = 0;
            argmax = -inf;
            for k='A':'B'
                likelihood = log(priors(k - 'A' + 1));
                for i=1:16
                    likelihood = likelihood + log(theta(i,testing(p,i), k - 'A' + 1));
                end
                if (argmax < likelihood)
                   argmax = likelihood;
                   best = k;
                end
            end

            classified(p) = best;
            
        end
        error = sum(classified ~= cell2mat(classesTesting)) / length(classified);
        error_avg(trial) = error;
        
        % logistic regression
        X = [ones(size(training, 1), 1) training];
        testing = [ones(length(testing), 1) testing];
        Y = cell2mat(classes);
        theta = zeros(1,17);

        % u = 1/(1+exp(-theta'*x_i))
        % P(y|x) = u(x)^y(1-u(x))^(1-y)
        delta = inf;
        steps = 0;
        % training
        while delta > EPSILON
            steps = steps + 1;
            theta_old = theta;
            for j=1:size(X,1)
                theta = theta + STEP_SIZE* ...
                    (Y(j)-'A' - 1/(1+exp(-theta*X(j,:)')))*X(j,:);
            end

            delta = sqrt(sum((theta_old - theta).^2));
        end

        % testing
        classified = zeros(length(testing), 1);
        for i=1:length(testing)
            classified(i) = round(1/(1+exp(-theta*testing(i,:)'))) + 'A';
        end
        errorLR = sum(classified ~= cell2mat(classesTesting)) / length(testing);
        error_avgLR(trial) = errorLR;
    end
    error_vs_training(m) = mean(error_avg);
    error_vs_trainingLR(m) = mean(error_avgLR);
    fprintf('%d of %d datasets(%d%%)\n', m, MAX_TRAINING, m/MAX_TRAINING*100)
end
error_vs_training(1) = error_vs_training(2);
error_vs_trainingLR(1) = error_vs_trainingLR(2);
close all
plot(error_vs_training)
hold all
plot(error_vs_trainingLR)
legend('Naive Bayes', 'Logistic Regression');
ylabel('Error Ratio (Misclassified / Total)');
xlabel('Training samples');