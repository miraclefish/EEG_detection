close all; clear all; clc;
str = 'C:\Users\Fish\EEG_detection\im\'; % ͼ��·��
for idx = 1:60  % ��ȡ5��ͼ��
    img{idx} = imread([str, num2str(idx), '.png']) ; % ����һ��cell����img{}�����ζ�ȡ��5��ͼ��
end

filename = 'hrc-2-pred.gif'; % �����gif��
for idx = 1:60  % for idx = 1:n
%  [A, map] = gray2ind(img{idx}, 256); 
   [A, map] = rgb2ind(img{idx}, 256);
   if idx == 1
      imwrite(A, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.2); % ��Ե�һ��ͼ��
      % ������ͼ�񱣴�Ϊvein.gif��LoopCount��ʾ�ظ������Ĵ�����Inf��LoopCountֵ��ʹ��������ѭ����DelayTimeΪ1����ʾ��һ֡ͼ����ʾ��ʱ��
   else % ��Ժ���ͼ��
      imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2); 
      % ������ͼ�񱣴�Ϊvein.gif��WriteMode��ʾд��ģʽ����append����overwrite�����ʹ�ã�appendģʽ�£�imwrite���������ļ���ӵ���֡
      % DelayTimeΪ1����ʾ����ͼ��Ĳ���ʱ����Ϊ1��
   end    
end