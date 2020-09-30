close all; clear all; clc;
str = 'C:\Users\Fish\EEG_detection\im\'; % 图像路径
for idx = 1:60  % 读取5幅图像
    img{idx} = imread([str, num2str(idx), '.png']) ; % 建立一个cell数组img{}，依次读取这5幅图像
end

filename = 'hrc-2-pred.gif'; % 保存的gif名
for idx = 1:60  % for idx = 1:n
%  [A, map] = gray2ind(img{idx}, 256); 
   [A, map] = rgb2ind(img{idx}, 256);
   if idx == 1
      imwrite(A, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.2); % 针对第一幅图像
      % 将索引图像保存为vein.gif，LoopCount表示重复动画的次数，Inf的LoopCount值可使动画连续循环，DelayTime为1，表示第一帧图像显示的时间
   else % 针对后续图像
      imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2); 
      % 将索引图像保存为vein.gif，WriteMode表示写入模式，与append（或overwrite）配合使用，append模式下，imwrite会向现有文件添加单个帧
      % DelayTime为1，表示后续图像的播放时间间隔为1秒
   end    
end