#!/usr/bin/octave -qf
arg_list = argv();
filename = arg_list{1};

%load('y_truth.mat');
%filename = 'NYU0001';
load(['../data/julia_data/Boxes/' filename '.mat']);
img=imread(['../data/SUNRGBD/kv1/NYUdata/' filename '/fullres/' filename '.jpg']);

_h = size(img,1);
_w = size(img,2);

function val = bound(x, lim)
  val = x;
  if x < 1 val=1; end
  if x > lim val=lim; end
endfunction

input2d = zeros(224,224,3,size(bounding, 2));
for i=1:size(bounding, 2)
  x=bounding(1,i)+1;
  y=bounding(2,i)+1;
  w=bounding(3,i);
  h=bounding(4,i);
  
  x=bound(x, _w);
  y=bound(y, _h);
  w=bound(w, _w);
  h=bound(h, _h);
  
  %rectangle('Position', [x y w h], 'EdgeColor','r');
  cropped = img(y:h-1, x:w-1, :);
  if size(cropped,1) > 1 && size(cropped, 2) > 1
    cropped=imresize(cropped, [224 224]);
    input2d(:,:,:, i)=cropped;
  end
end

clear _h
clear _w
clear ans
clear arg_list
clear cropped
clear filename
clear h
clear i
clear img
clear w
clear x
clear y


input2d1=input2d(:,:,:, 1:500);
input2d2=input2d(:,:,:, 501:size(bounding, 2));

clear bounding
clear input2d


save('../data/julia_data/temp.mat', '-v7');
