% Demo for Structured Edge Detector (please see readme.txt first).
addpath('./toolbox/');
%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval(model, 'show',1, 'name','' ); end

img_dir = 'D:\VGGFace2\test_all\';
edge_dir = 'D:\VGGFace2\test_all_128\edges\';
resize_img_dir = 'D:\VGGFace2\test_all_128\imgs\';
sub_dir = dir(img_dir);
t3 = clock;
for i = 1:length(sub_dir)
    if(isequal(sub_dir(i).name,'.')||isequal(sub_dir(i).name,'..') ...
            ||~sub_dir(i).isdir)   %如果不是目录则跳过
        continue;
    end
    sub_dirpath = fullfile(img_dir,sub_dir(i).name,'*.jpg');
    jpg_files = dir(sub_dirpath); %子文件夹下找后缀为jpg的文件
    sub_edge_dir = strcat(edge_dir,sub_dir(i).name);
    sub_resize_dir = strcat(resize_img_dir,sub_dir(i).name);
    if ~exist(sub_resize_dir,'dir')
        mkdir(sub_resize_dir);
    end

    if ~exist(sub_edge_dir,'dir')
        mkdir(sub_edge_dir);
    end
    t1 = clock;
    for j = 1:length(jpg_files)
        jpg_path = fullfile(img_dir,sub_dir(i).name,jpg_files(j).name);
        [pathstr,name,suffix] = fileparts(jpg_path);
        %边缘处理
        I = imresize(imread(jpg_path),[256,256]);
        resize_img = imresize(I,[128,128]);
        imwrite(resize_img,[sub_resize_dir,'\',name,'.png'])
        E=edgesDetect(I,model);
        E1 = imresize(E,[128,128]);
        [Ox,Oy] = gradient2(convTri(E1,4));
        [Oxx,~] = gradient2(Ox);
        [Oxy,Oyy] = gradient2(Oy);
        O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
        E2 = edgesNmsMex(E1,O,1,5,1.01,1);
        E3 = double(E2>=max(eps,20.0/255.0));
        E3 = bwmorph(E3,'thin',inf);
        E4 = bwareaopen(E3, 3);
        E4=1-E4;
        E_simple = uint8(E4*255);
        imwrite(E_simple, [sub_edge_dir,'\', name, '.bmp']);
    end
    t2=clock;
    spend_time=etime(t2,t1);
    disp([sub_dirpath,'运行时间：',num2str(spend_time)]);
end
t4 = clock;
total_time = etime(t4,t3);
disp(['总时间为：',num2str(total_time)]);

% fileList = dir(root_dir);
% n = length(fileList);
% for i = 1:n
%     if strcmp(fileList(i).name,'.')==1||strcmp(fileList(i).name,'..')==1
%         continue;
%     else
%         file = fileList(i).name;
%         [pathstr,name,suffix] = fileparts(file);
%         I = imresize(imread([root_dir, file]),[512,512]);
%         E=edgesDetect(I,model);
%         E1 = imresize(E,[256,256]);
%         [Ox,Oy] = gradient2(convTri(E1,4));
%         [Oxx,~] = gradient2(Ox);
%         [Oxy,Oyy] = gradient2(Oy);
%         O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
%         E2 = edgesNmsMex(E1,O,1,5,1.01,1);
%         E3 = double(E2>=max(eps,20.0/255.0));
%         E3 = bwmorph(E3,'thin',inf);
%         E4 = bwareaopen(E3, 10);
%         E4=1-E4;
%         E_simple = uint8(E4*255);
%         imwrite(E_simple, ['D:\VGGFace2\VGG-Face2\data\vggface2_train.tar\vggface2_train\edges\n000002\', name, '.bmp']);
%     end
% end
       


% filename = '2';
% I = imresize(imread(['..\data\imgs\', filename, '.png']),[512,512]);
% E=edgesDetect(I,model);
% E1 = imresize(E,[256,256]);
% [Ox,Oy] = gradient2(convTri(E1,4));
% [Oxx,~] = gradient2(Ox);
% [Oxy,Oyy] = gradient2(Oy);
% O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
% E2 = edgesNmsMex(E1,O,1,5,1.01,1);

% E3 = double(E2>=max(eps,20.0/255.0));
% E3 = bwmorph(E3,'thin',inf);
% E4 = bwareaopen(E3, 10);
% E4=1-E4;
% E_simple = uint8(E4*255);
% imwrite(E_simple, ['..\data\edges\', filename, '.bmp']); 