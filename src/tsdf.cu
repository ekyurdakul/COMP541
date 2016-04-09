//Compilation:
//nvcc -std=c++11 -o tsdf tsdf.cu

#include "tsdf_base.h"

/*
{
	"type": "Scene3DData",
	"name": "dataTest",
	"phase": "Testing",
	"data_root": "/home/shurans/deepDetectLocal/sunrgbd_dss_data/",
	"file_list": "/n/fs/modelnet/deepDetect/Release/result/proposal//RPN_NYU//boxes_NYU_po_test_nb2000_fb.list",
	"grid_size": [3,30,30,30],
	"batch_size": [288,96],
	"num_categories": 20,
	"bb_param_weight": [1,1,1,1,1,1],
	"encode_type": 100,
	"scale": 100,
	"box_reg": true,
	"context_pad": 3,
	"out": [
		"data",
		"label",
		"bb_tar_diff",
		"bb_loss_weights"
	],
	"GPU": 0,
	"num_percate": 0,
	"is_render": false,
	"is_combineimg": false,
	"is_combinehha": false,
	"img_fea_folder": "/home/shurans/deepDetectLocal/image_fea/RPN_NYU/po/",
	"imgfea_dim": 4096,
	"box_2dreg": false,
	"orein_cls": false,
}
*/

void compute_TSDF(string binfile, Box3D box, StorageT* datamem, vector<int> grid_size, int encode_type, float scale) {
	//compute_TSDF(box, dataGPUmem, grid_size,encode_type,scale);
    float tsdf_size = grid_size[1];
    if (grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]){
        cerr << "grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]" << endl;
        exit(EXIT_FAILURE);
    }
    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((tsdf_size*tsdf_size*tsdf_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    float* bb3d_data;
    cudaMalloc(&bb3d_data,  sizeof(float)*15);
    

//Scene


    int filesize =0;
	string filename=binfile;
      std::cout<< "loading image "<< filename<<std::endl;
/*

      FILE* fp = fopen(filename.c_str(),"rb");
      if (fp==NULL) { std::cout<<"in load() :fail to open file: "<<filename<<std::endl; exit(EXIT_FAILURE); }
      grid_range = new unsigned int[3];
      filesize += fread((void*)(grid_range), sizeof(unsigned int), 3, fp);
      
      begin_range = new float[3];
      filesize += fread((void*)(begin_range), sizeof(float), 3, fp);
      filesize += fread((void*)(&grid_delta), sizeof(float), 1, fp);

      RGBDimage = new RGBDpixel[width*height];
      filesize += fread((void*)(RGBDimage), sizeof(RGBDpixel), width*height, fp);

      filesize +=  fread((void*)(&len_beIndex), sizeof(unsigned int), 1, fp);
      beIndex   = new unsigned int [len_beIndex];
      filesize += fread((void*)(beIndex), sizeof(unsigned int), len_beIndex, fp);

      filesize +=  fread((void*)(&len_beLinIdx), sizeof(unsigned int), 1, fp);
      beLinIdx  = new unsigned int [len_beLinIdx];
      filesize += fread((void*)(beLinIdx), sizeof(unsigned int), len_beLinIdx, fp);

      filesize += fread((void*)(&len_pcIndex), sizeof(unsigned int), 1, fp);
      pcIndex   = new unsigned int [len_pcIndex];
      filesize += fread((void*)(pcIndex), sizeof(unsigned int), len_pcIndex, fp);
      fclose(fp);

      GPUdata = false;
/*
    if (!GPUdata){
       if (beIndex!=NULL){
           unsigned int* beIndexCPU = beIndex;
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMalloc(&beIndex, sizeof(unsigned int)*len_beIndex));
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMemcpy(beIndex, beIndexCPU,sizeof(unsigned int)*len_beIndex, cudaMemcpyHostToDevice));
           delete [] beIndexCPU;
       }
       else{
           std::cout << "beIndex is NULL"<<std::endl;
       }

       if (beLinIdx!=NULL){
           unsigned int* beLinIdxCPU = beLinIdx;
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMalloc(&beLinIdx, sizeof(unsigned int)*len_beLinIdx));
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMemcpy(beLinIdx, beLinIdxCPU,sizeof(unsigned int)*len_beLinIdx, cudaMemcpyHostToDevice));
           delete [] beLinIdxCPU;
       }
       else{
           std::cout << "beLinIdx is NULL"<<std::endl;
       }

       // make it to full matrix to skip searching 
       unsigned int * beIndexFull;
       unsigned int sz = 2*sizeof(unsigned int)*(grid_range[0]+1)*(grid_range[1]+1)*(grid_range[2]+1);
       checkCUDA(__LINE__, cudaMalloc(&beIndexFull, sz));
       checkCUDA(__LINE__, cudaMemset(beIndexFull, 0, sz));
       int THREADS_NUM = 1024;
       int BLOCK_NUM = int((len_beLinIdx + size_t(THREADS_NUM) - 1) / THREADS_NUM);
       fillInBeIndexFull<<<BLOCK_NUM,THREADS_NUM>>>(beIndexFull,beIndex,beLinIdx,len_beLinIdx);
       checkCUDA(__LINE__,cudaGetLastError());
       checkCUDA(__LINE__, cudaFree(beIndex));      beIndex = NULL;
       checkCUDA(__LINE__, cudaFree(beLinIdx));     beLinIdx = NULL;
       beIndex = beIndexFull;

       if (pcIndex!=NULL){
          unsigned int* pcIndexCPU = pcIndex;
          checkCUDA(__LINE__, cudaMalloc(&pcIndex, sizeof(unsigned int)*len_pcIndex));
          checkCUDA(__LINE__, cudaMemcpy(pcIndex, pcIndexCPU,sizeof(unsigned int)*len_pcIndex, cudaMemcpyHostToDevice));
          delete [] pcIndexCPU;
       }
       else{
           std::cout << "pcIndexCPU is NULL"<<std::endl;
       }
       

       if (RGBDimage!=NULL){
         RGBDpixel* RGBDimageCPU = RGBDimage;
         checkCUDA(__LINE__, cudaMalloc(&RGBDimage, sizeof(RGBDpixel)*width*height));
         checkCUDA(__LINE__, cudaMemcpy( RGBDimage, RGBDimageCPU, sizeof(RGBDpixel)*width*height, cudaMemcpyHostToDevice));
         delete [] RGBDimageCPU;
       }
       else{
           std::cout << "RGBDimage is NULL"<<std::endl;
       }

       if (grid_range!=NULL){ 
          unsigned int * grid_rangeCPU = grid_range;
          checkCUDA(__LINE__, cudaMalloc(&grid_range, sizeof(unsigned int)*3));
          checkCUDA(__LINE__, cudaMemcpy(grid_range, grid_rangeCPU, 3*sizeof(unsigned int), cudaMemcpyHostToDevice));
          delete [] grid_rangeCPU;
       }
       else{
          std::cout << "grid_range is NULL"<<std::endl;
       }

       if (begin_range!=NULL){ 
          float * begin_rangeCPU = begin_range;
          checkCUDA(__LINE__, cudaMalloc(&begin_range, sizeof(float)*3));
          checkCUDA(__LINE__, cudaMemcpy(begin_range, begin_rangeCPU, sizeof(float)*3, cudaMemcpyHostToDevice));
          delete [] begin_rangeCPU;
       }
       else{
          std::cout << "grid_range is NULL"<<std::endl;
       }


       checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)K, sizeof(float)*9, cudaMemcpyHostToDevice));

      
       checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R, sizeof(float)*9, cudaMemcpyHostToDevice)); 

       GPUdata = true;

    }



if (!GPUdata){
         std::cout<< "Data is not at GPU cannot compute_xyz at GPU"<<std::endl;
         FatalError(__LINE__);
      }
      if (XYZimage!=NULL){
         std::cout<< "XYZimage!=NULL"<<std::endl;
         FatalError(__LINE__);
      }
      checkCUDA(__LINE__, cudaMalloc(&XYZimage, sizeof(float)*width*height*3));
      compute_xyzkernel<<<width,height>>>(XYZimage,RGBDimage,K_GPU,R_GPU);
*/
	cudaMemcpy(bb3d_data, box.base, sizeof(float)*15, cudaMemcpyHostToDevice);

/*
	unsigned int * grid_range = scene->grid_range;
	float* R_data = scene->R_GPU;
	float* K_data = scene->K_GPU;
	float* range  = scene->begin_range;

	RGBDpixel* RGBDimage = scene->RGBDimage;
	unsigned int* star_end_indx_data = scene->beIndex;
	unsigned int* pc_lin_indx_data = scene->pcIndex;
	float* XYZimage  = scene->XYZimage;

	 if (encode_type > 99){
	    compute_TSDFGPUbox<<<BLOCK_NUM,THREADS_NUM>>>(datamem, R_data, K_data, range, scene->grid_delta, grid_range, RGBDimage, 
		           star_end_indx_data, pc_lin_indx_data, XYZimage, bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
		           scene->width, scene->height, encode_type, scale);

	}
	else{
	  //compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(datamem, R_data, K_data, RGBDimage, XYZimage, bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], scene->width, scene->height, encode_type, scale);
	}
*/
	cudaDeviceSynchronize();
	cudaGetLastError();
    cudaFree(bb3d_data);
}



int main(){
	//boxes_NYU_po_test_nb2000_fb.list
	string file_list = "boxes_SUNrgbd_po_test_nb2000_fb.list";
    	string data_root =  "";

    	float scale =100;
	float context_pad =3;
	vector<int> grid_size {3,30,30,30};
    	int encode_type =100;

    	cout <<"loading file "<< file_list << endl;
    	FILE* fp = fopen(file_list.c_str(),"rb");
    	if (fp==NULL) { cout<< "failed to open file: "<< file_list << endl; exit(EXIT_FAILURE); }


	unsigned int totalScenes = 0;
	unsigned int totalBoxes = 0;
    while (feof(fp)==0) {
	totalScenes++;
	
      unsigned int len = 0;
      fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
	string filename = "";
      	filename.resize(len);
      if (len>0) fread((void*)(filename.data()), sizeof(char), len, fp);


	string binfile = data_root+filename+".bin";
	string tsdffile = data_root+filename+".tsdf";
	float R[9];
	float K[9];
	float height;
	float width;
      	fread((void*)(R), sizeof(float), 9, fp);
 	fread((void*)(K), sizeof(float), 9, fp);
      	fread((void*)(&height), sizeof(unsigned int), 1, fp);  
      fread((void*)(&width), sizeof(unsigned int), 1, fp); 
      

      fread((void*)(&len),    sizeof(unsigned int),   1, fp);
      if (len>0){
	FILE * fid = fopen(tsdffile.c_str(),"wb");
	cout << binfile << " " << totalScenes << " ";
          for (int i=0;i<len;++i){
		totalBoxes++;

              Box3D box;
              fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
              fread((void*)(box.base),        sizeof(float), 9, fp);
              fread((void*)(box.center),      sizeof(float), 3, fp);
              fread((void*)(box.coeff),       sizeof(float), 3, fp);
              box = processbox (box, context_pad, grid_size[1]);

		//Compute TSDF for each box
		StorageT* dataGPUmem;
		float* dataCPUmem = new float[3*30*30*30];
		cudaMalloc(&dataGPUmem, 3*30*30*30*sizeof(float));

		compute_TSDF(binfile, box, dataGPUmem, grid_size,encode_type,scale);

		cudaMemcpy(dataCPUmem, dataGPUmem,3*30*30*30*sizeof(float), cudaMemcpyDeviceToHost);
	    	fwrite(dataCPUmem,sizeof(float),3*30*30*30,fid);
		cudaFree(dataGPUmem);
		delete[] dataCPUmem;
             
          }
	cout << totalBoxes << endl;
    	fclose(fid);
      }
    }
    fclose(fp);
	return 1;
}
