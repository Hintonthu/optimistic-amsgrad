adam = load('./mnist_rnd_base.mat')
optadam_rna =  load('./mnist_rnd_opt.mat')

xData = 1:length(adam.test_loss);
yData{1} = adam.test_loss;
yData{2} = optadam_rna.test_loss;
legendStr = {'AMSGrad','Opt-AMSGrad (r=5)'};

%% Pretty Plot

figure;
options.logScale = 0;
options.colors = [1 0 0
    0 1 0
    0 0 1];
options.lineStyles = {'-','-','-'};
options.xlabel = 'epoch';
options.ylabel = 'testing loss (cross entropy)';
options.legend = legendStr;
options.legendLoc = 'NorthEast';
prettyPlot(xData,yData,options);

clearvars xData;
clearvars yData;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xData = 1:length(adam.test_acc);
yData{1} = adam.test_acc;
yData{2} = optadam_rna.test_acc;
legendStr = {'AMSGrad','Opt-AMSGrad (r=5)'};

%% Pretty Plot

figure;
options.logScale = 0;
options.colors = [1 0 0
    0 1 0
    0 0 1];
options.lineStyles = {'-','-','-'};

options.xlabel = 'epoch';
options.ylabel = 'testing classfication accuracy (%)';
options.legend = legendStr;
options.legendLoc = 'SouthEast';
prettyPlot(xData,yData,options);

clearvars xData;
clearvars yData;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

yData{1} = adam.train_loss(:,end);
yData{2} = optadam_rna.train_loss(:,end);
xData = 1:length(yData{1});

%yData{1} = reshape(adam.train_loss',[],1);
%yData{2} = reshape(optadam_rna.train_loss',[],1);
%xData{1} = 1.0/19:1.0/19:10
%xData{2} = 1.0/19:1.0/19:10
legendStr = {'AMSGrad','Opt-AMSGrad (r=5)'};

%% Pretty Plot

figure;
options.logScale = 0;
options.colors = [1 0 0
    0 1 0
    0 0 1];
options.lineStyles = {'-','-','-'};
options.logScale = 2;
options.xlabel = 'epoch';
options.ylabel = 'traing loss (cross entropy)';
options.legend = legendStr;
options.legendLoc = 'NorthEast';
prettyPlot(xData,yData,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

yData{1} = adam.train_acc(:,end);
yData{2} = optadam_rna.train_acc(:,end);
xData = 1:length(yData{1});

%yData{1} = reshape(adam.train_acc',[],1);
%yData{2} = reshape(optadam_rna.train_acc',[],1);
%xData{1} = 1.0/19:1.0/19:10
%xData{2} = 1.0/19:1.0/19:10
legendStr = {'AMSGrad','Opt-AMSGrad (r=5)'};

%% Pretty Plot

figure;
options.logScale = 0;
options.colors = [1 0 0
    0 1 0
    0 0 1];
options.lineStyles = {'-','-','-'};

options.xlabel = 'epoch';
options.ylabel = 'traing classification accuracy (%)';
options.legend = legendStr;
options.legendLoc = 'SouthEast';
prettyPlot(xData,yData,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
