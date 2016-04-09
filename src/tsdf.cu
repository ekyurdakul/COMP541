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


void compute_TSDF(std::vector<Scene3D*> *chosen_scenes_ptr, std::vector<int> *chosen_box_id, StorageT* datamem, std::vector<int> grid_size, int encode_type, float scale) {
/*
    // for each scene 
    int totalcounter = 0;
    float tsdf_size = grid_size[1];
    if (grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]){
        std::cerr << "grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]" <<std::endl;
        exit(EXIT_FAILURE);
    }

    int numeltsdf = grid_size[0]*tsdf_size*tsdf_size*tsdf_size;
    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((tsdf_size*tsdf_size*tsdf_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    float* bb3d_data;

    //int tmpD; cudaGetDevice(&tmpD); std::cout<<"GPU at LINE "<<__LINE__<<" = "<<tmpD<<std::endl;
    //checkCUDA(__LINE__,cudaDeviceSynchronize());
    checkCUDA(__LINE__, cudaMalloc(&bb3d_data,  sizeof(float)*15));
    
    //unsigned long long transformtime =0;
    //unsigned long long loadtime =0;
    //unsigned long long copygputime =0;
    //unsigned int sz = 0;
    Scene3D* scene_prev = NULL;
    for (int sceneId = 0;sceneId<(*chosen_scenes_ptr).size();sceneId++){
        // caculate in CPU mode
        //compute_TSDFCPUbox(tsdf_data,&((*chosen_scenes_ptr)[sceneId]),boxId,grid_size,encode_type,scale);
        // caculate in GPU mode
        
        //unsigned long long  time0,time1,time2,time3,time4;
        Scene3D* scene = (*chosen_scenes_ptr)[sceneId];
        //int tmpD; cudaGetDevice(&tmpD); std::cout<<"GPU at LINE "<<__LINE__<<" = "<<tmpD<<std::endl;
        // perpare scene
        if (scene!=scene_prev){
            if (scene_prev!=NULL){
               scene_prev -> free();
            }
            scene->loadData2XYZimage(); 
        }
        
        int boxId = (*chosen_box_id)[sceneId];
        checkCUDA(__LINE__, cudaMemcpy(bb3d_data, scene->objects[boxId].base, sizeof(float)*15, cudaMemcpyHostToDevice));

        unsigned int * grid_range = scene->grid_range;
        float* R_data = scene->R_GPU;
        float* K_data = scene->K_GPU;
        float* range  = scene->begin_range;
        
        RGBDpixel* RGBDimage = scene->RGBDimage;
        unsigned int* star_end_indx_data = scene->beIndex;
        unsigned int* pc_lin_indx_data = scene->pcIndex;
        float* XYZimage  = scene->XYZimage;
        
        // output
        StorageT * tsdf_data = &datamem[totalcounter*numeltsdf];

        //time3 = get_timestamp_dss();   
        //checkCUDA(__LINE__,cudaDeviceSynchronize());
         if (encode_type > 99){
            compute_TSDFGPUbox<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data, R_data, K_data, range, scene->grid_delta, grid_range, RGBDimage, 
                           star_end_indx_data, pc_lin_indx_data, XYZimage, bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                           scene->width, scene->height, encode_type, scale);

        }
        else{
          //std::cout<<"compute_TSDFGPUbox_proj"<<std::endl;
          compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data, R_data, K_data, RGBDimage, XYZimage,
                                                             bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                                                             scene->width, scene->height, encode_type, scale);
        }
        
        checkCUDA(__LINE__,cudaDeviceSynchronize());
        checkCUDA(__LINE__,cudaGetLastError());
        //time4 = get_timestamp_dss();

        //

        ++totalcounter;

        scene_prev = scene;
        //loadtime += time1-time0;
        //copygputime += time2-time1;
        //transformtime += time4-time3;
    }
    checkCUDA(__LINE__, cudaFree(bb3d_data));
    
    // free the loaded images
    for (int sceneId = 0;sceneId<(*chosen_scenes_ptr).size();sceneId++){
        (*chosen_scenes_ptr)[sceneId]->free();
    }
    
    
    //std::cout << "compute_TSDF: read disk " << loadtime/1000 << " ms, " << "copygputime " 
    //<< copygputime/1000 << "transform " << transformtime/1000 << " ms" <<std::endl;  
*/
}



int main(){
/*
std::string file_list = "DSS/boxfile/boxes_NYU_trainfea_debug.list";
    //std::string data_root = "DSS/sunrgbd_dss_data/";
    std::string data_root =  "/n/fs/modelnet/deepDetect/sunrgbd_dss_data/";
    std::vector<Scene3D*> scenes;

    //int count = 0;
    int object_count = 0;
    float scale =100;
    float context_pad =3;
    std::vector<int> grid_size {3,30,30,30};
    int encode_type =100;

    std::cout<<"loading file "<<file_list<<"\n";
    FILE* fp = fopen(file_list.c_str(),"rb");
    if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }
    while (feof(fp)==0) {
      Scene3D* scene = new Scene3D();
      unsigned int len = 0;
      fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
      scene->filename.resize(len);
      if (len>0) fread((void*)(scene->filename.data()), sizeof(char), len, fp);
      scene->filename = data_root+scene->filename+".bin"; 
      fread((void*)(scene->R), sizeof(float), 9, fp);
      fread((void*)(scene->K), sizeof(float), 9, fp);
      fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);  
      fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
      

      fread((void*)(&len),    sizeof(unsigned int),   1, fp);
      scene->objects.resize(len);
      if (len>0){
          for (int i=0;i<len;++i){
              Box3D box;
              fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
              fread((void*)(box.base),        sizeof(float), 9, fp);
              fread((void*)(box.center),      sizeof(float), 3, fp);
              fread((void*)(box.coeff),       sizeof(float), 3, fp);
              //process box pad contex oreintation 
              box = processbox (box, context_pad, grid_size[1]);
              scene->objects[i]=box;

              object_count++;
              //num_categories = max(num_categories, box.category);
            
              //printf("category:%d\n",box.category);
              //printf("box.base:%f,%f,%f,%f,%f,%f\n",box.base[0],box.base[1],box.base[2],box.base[3],box.base[4],box.base[5]);
              //printf("box.base:%f,%f,%f,%f,%f,%f\n",box.base[0],box.base[1],box.base[2],box.base[3],box.base[4],box.base[5]);
              //printf("box.center:%f,%f,%f\n",box.center[0],box.center[1],box.center[2]);
              //printf("box.coeff:%f,%f,%f\n",box.coeff[0],box.coeff[1],box.coeff[2]);
             
          }
      }
      scenes.push_back(scene);

    }
    fclose(fp);

    std::vector<Scene3D*> chosen_scenes;
    std::vector<int> chosen_box_id;
    for (int i = 0;i<scenes.size();++i){
       for (int j =0; j < scenes[i]->objects.size();++j){
            chosen_scenes.push_back(scenes[i]);
            chosen_box_id.push_back(j);
       } 
    }

    
    std::cout<<"object_count:" <<object_count <<std::endl;
    float* dataCPUmem = new float[(object_count)*3*30*30*30];
    StorageT* dataGPUmem;
    checkCUDA(__LINE__, cudaMalloc(&dataGPUmem, (object_count)*3*30*30*30*sizeof(float)));

    compute_TSDF(&chosen_scenes, &chosen_box_id, dataGPUmem,grid_size,encode_type,scale);
    checkCUDA(__LINE__, cudaMemcpy(dataCPUmem, dataGPUmem,(object_count)*3*30*30*30*sizeof(float), cudaMemcpyDeviceToHost) );
        

    std::string outputfile = "DSS/feature.bin";

    FILE * fid = fopen(outputfile.c_str(),"wb");
    fwrite(dataCPUmem,sizeof(float),(object_count)*3*30*30*30,fid);
    fclose(fid);
    return 1;
*/
	//string file_list = "..//data//boxes_NYU_po_test_nb2000_fb.list";
	string file_list = "..//data//boxes_SUNrgbd_po_test_nb2000_fb.list";
    	string data_root =  "..//data//";
	string output_data = "..//data//julia_data//";

    	float scale =100;
	float context_pad =3;
	vector<int> grid_size {3,30,30,30};
    	int encode_type =100;

    	cout <<"loading file "<< file_list << endl;
    	FILE* fp = fopen(file_list.c_str(),"rb");
    	if (fp==NULL) { cout<< "failed to open file: "<< file_list << endl; exit(EXIT_FAILURE); }

	unsigned int totalScenes = 0;
	unsigned int totalBoxes = 0;

	//limit to 5 scenes for testing
	int maxscenes = 5;

    	while (feof(fp)==0 && totalScenes < maxscenes) {	
	      	unsigned int len = 0;
	      	fread((void*)(&len), sizeof(unsigned int), 1, fp);    
	      	if (len==0) break;
		string filename = "";
	      	filename.resize(len);
	      	if (len>0) fread((void*)(filename.data()), sizeof(char), len, fp);

		int lastback = filename.find_last_of("/");
		string outputname = "";
		if (lastback > 0)
		{
			outputname = filename.substr(lastback+1);
		}
		else continue;
		string binfile = data_root+filename+".bin";
		string tsdffile = output_data+outputname+".tsdf";		

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

			cout << totalScenes+1 << ": Boxes :" << len << " " << "TSDF\t" << tsdffile << "\t" << "Bin\t" << binfile << endl;

			FILE * fid = fopen(tsdffile.c_str(),"wb");
			for (int i=0;i<len;++i){
				totalBoxes++;

			      	Box3D box;
			      	fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
			      	fread((void*)(box.base),        sizeof(float), 9, fp);
			      	fread((void*)(box.center),      sizeof(float), 3, fp);
			      	fread((void*)(box.coeff),       sizeof(float), 3, fp);
			      	box = processbox (box, context_pad, grid_size[1]);

				StorageT* dataGPUmem;
				float* dataCPUmem = new float[3*30*30*30];
				cudaMalloc(&dataGPUmem, 3*30*30*30*sizeof(float));

				//Compute TSDF for each box and write it to file
				compute_TSDF(binfile, box, dataGPUmem, grid_size,encode_type,scale);

				cudaMemcpy(dataCPUmem, dataGPUmem,3*30*30*30*sizeof(float), cudaMemcpyDeviceToHost);
			    	fwrite(dataCPUmem,sizeof(float),3*30*30*30,fid);
				cudaFree(dataGPUmem);
				delete[] dataCPUmem;
             
          		}//for boxes
    			fclose(fid);
      		}

		totalScenes++;
    	}//while feof scenes

	fclose(fp);
	return 1;
}
