%% erik reed
% kmeans-clustering

% scatter(data(:,1),data(:,2), 35, [127,127,127]/255)
% hold all

clear all; 

data = load('cluster1.csv');
kmeansPP = true;
kmax=15;
runs = 5;

for k=1:kmax

[dimX, dimY] = size(data);
runs_min = zeros(runs, 1);
runs_mean = zeros(runs, 1);
runs_std = zeros(runs, 1);

for run=1:runs
    clusters = zeros(k, dimY);
    if kmeansPP && k ~= 1
        % K-means++ for initialization
        clusters(1,:) = data(randi(dimX,1,1), :);
        for i=2:k
            distances = zeros(dimX, 1);
            for j=1:dimX
                minDist = inf;
                for l=1:i-1
                    dist = norm(clusters(l, :) - data(j, :));
                    minDist = min(minDist, dist);
                end
                distances(j) = minDist;
            end
            probs = [cumsum(distances/sum(distances))];
            nextCluster = find(probs >= rand(), 1);
            clusters(i,:) = data(nextCluster, :);
        end
    else
        % Lloyd's algorithm for initialization
        clusters(1:k,:) = data(randi(dimX,k,1), :);
    end
    
    memberships = zeros(dimX, 1);
    distances = zeros(dimX, 1);
    
    delta = inf;
    iters = 0;
    while delta > 0
        old = memberships;
        for i=1:dimX
            newCluster = 1;
            minDist = norm(clusters(1, :) - data(i, :));
            for j=2:k
                dist = norm(clusters(j, :) - data(i, :));
                if dist < minDist
                    newCluster = j;
                    minDist = dist;
                end
            end
            memberships(i) = newCluster;
            distances(i) = minDist;
        end
        iters = iters + 1;
        delta = sum(old ~= memberships);
        
        % update cluster centers
        for i=1:k
            clusters(i,:) = sum(data(memberships == i, :)) / ... 
                size(data(memberships == i, :),1);
        end
        
    end
    runs_min(run) = min(distances.^2);
    runs_mean(run) = mean(distances.^2);
    runs_std(run) = std(distances.^2);
%     scatter(clusters(:,1),clusters(:,2), [], [0,0,0], 'filled')
end

%%
subplot(3,1,1)
scatter(1:200, runs_min)
ylabel('min')
subplot(3,1,2)
scatter(1:200, runs_mean)
ylabel('mean')
subplot(3,1,3)
scatter(1:200, runs_std)
ylabel('stdev')
xlabel('runs')
