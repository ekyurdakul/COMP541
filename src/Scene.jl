using Images;
using MAT;

function bound(x, lim)
	val = x;
	if x < 1 val=1; end
	if x > lim val=lim; end
	return val;
end

function prepareScene(filename)
	matfile = matread("../data/julia_data/Boxes/$filename.mat");
	img=load("../data/SUNRGBD/kv1/NYUdata/$filename/fullres/$filename.jpg");

	_h = size(img,1);
	_w = size(img,2);

	bounding = matfile["bounding"];

	input2d = zeros(224,224,3,size(bounding, 2));
	for i=1:size(bounding, 2)
		x=bounding[1,convert(Int32,i)];
		y=bounding[2,convert(Int32,i)];
		w=bounding[3,convert(Int32,i)];
		h=bounding[4,convert(Int32,i)];

		x=bound(y, _h);
		y=bound(x, _w);
		w=bound(h, _h);
		h=bound(w, _w);

		x=convert(Int32, x);
		y=convert(Int32, y);
		w=convert(Int32, w);
		h=convert(Int32, h);

		cropped = img[x:w, y:h];
		if size(cropped,1) >= 1 && size(cropped, 2) >= 1
			cropped=Images.imresize(cropped, (224,224));
			input2d[:,:,:, i]=separate(cropped);
		end
	end
	return input2d, size(bounding, 2);
end
