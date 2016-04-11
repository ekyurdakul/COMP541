//Include
#include "tsdf_base.h"

//Global variables
std::vector<Scene3D*> scenes;
std::vector<int> box_id;
int totalObjectCount = 0;
float scale = 100;
float context_pad =3;
std::vector<int> grid_size {3,30,30,30};
int encode_type =100;
int totalScenes = 0;
string file_list = "";
string data_root =  "..//data//";
string output_data = "..//data//julia_data//";
int maxscenes = 0;
//list file
FILE* fp = NULL;

//Main function
int main(){
	cout << "Loaded my custom library." << endl;
	//New load 2d boxes
    int counter = 0;
	string file_list = data_root+"boxes2d_NYU_po_nb2000.list";

    cout<<"loading file "<<file_list<<"\n";
    FILE* fp = fopen(file_list.c_str(),"rb");
    if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }

/*
    while (feof(fp)==0) {
      Scene3D* scene = new Scene3D();
      unsigned int len = 0;
      size_t file_size = 0;
      file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
      scene->filename.resize(len);
      if (len>0) file_size += fread((void*)(scene->filename.data()), sizeof(char), len, fp);
     

      file_size += fread((void*)(scene->R), sizeof(float), 9, fp);
      file_size += fread((void*)(scene->K), sizeof(float), 9, fp);
      file_size += fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);
      file_size += fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
      file_size += fread((void*)(&len),    sizeof(unsigned int),   1, fp);
      scene->objects.resize(len);
      
      //std::cout<<scene->filename <<std::endl; 
      //std::cout<<len <<std::endl; 
      //std::cin.ignore();
      for (int bid = 0;bid<len;bid++){
	 Box2D box;
	 file_size += fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
	 file_size += fread((void*)(box.tblr),        sizeof(float), 4, fp);
	 scene->objects_2d_tight.push_back(box);
	 
	 uint8_t hasTarget = 0;
	 file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp);
	 if (hasTarget>0){
	  std::cout<<" sth wrong in line "   << __LINE__ << std::endl;
	 }

	 file_size += fread((void*)(box.tblr),   sizeof(float), 4, fp);
	 scene->objects_2d_full.push_back(box);
	 file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp);
	 if (hasTarget>0){
	  std::cout<<" sth wrong in line "   << __LINE__ << std::endl;
	 }
      }
      scenes.push_back(scene);
    }
    fclose(fp);
*/
	return 1;
}

//Export functions to be called from Julia
extern "C"
{
	//Initialize variables
	void initTSDF(int db, int maxs)
	{
		if (db == 0)
			file_list = "..//data//boxes_NYU_po_test_nb2000_fb.list";
		else
			file_list = "..//data//boxes_SUNrgbd_po_test_nb2000_fb.list";

		cout << "Loading file: " << file_list << endl << endl;
		fp = fopen(file_list.c_str(),"rb");
		if (fp==NULL) { cout << "Failed to open file: "<< file_list << endl; exit(EXIT_FAILURE); }
		maxscenes = maxs;
	}

	//Call this when done with TSDF
	void freeTSDF()
	{
		fclose(fp);
	}

	//Compute TSDF for the next scene
	int getNextTSDF(int& count)
	{
		if (feof(fp)==0 && totalScenes < maxscenes)
		{
			Scene3D* scene = new Scene3D();
			unsigned int len = 0;
			fread((void*)(&len), sizeof(unsigned int), 1, fp);    
			if (len==0) return -1;
			scene->filename.resize(len);
			if (len>0) fread((void*)(scene->filename.data()), sizeof(char), len, fp);

			//TSDF file
			string tsdffile = output_data+"temp.tdsf";
			scene->filename = data_root+scene->filename+".bin";

			fread((void*)(scene->R), sizeof(float), 9, fp);
			fread((void*)(scene->K), sizeof(float), 9, fp);
			fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);  
			fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 


			fread((void*)(&len),    sizeof(unsigned int),   1, fp);
			scene->objects.resize(len);
			if (len>0){
			  totalObjectCount += len;
			  for (int i=0;i<len;++i){
			      Box3D box;
			      fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
			      fread((void*)(box.base),        sizeof(float), 9, fp);
			      fread((void*)(box.center),      sizeof(float), 3, fp);
			      fread((void*)(box.coeff),       sizeof(float), 3, fp);
			      //process box pad contex oreintation 
			      box = processbox (box, context_pad, grid_size[1]);
			      scene->objects[i]=box;
			      box_id.push_back(i);
			  }
			}
			scenes.push_back(scene);
			totalScenes++;

			cout << "Scene: " << totalScenes << " Boxes: " << len << " Bin: " << scene->filename << " TSDF: " << tsdffile << endl << endl;

			unsigned long long  time0,time1;

			time0 = get_timestamp_dss();
			float* dataCPUmem = new float[len*3*30*30*30];
			StorageT* dataGPUmem;
			checkCUDA(__LINE__, cudaMalloc(&dataGPUmem, (len)*3*30*30*30*sizeof(float)));
			time1 = get_timestamp_dss();
			cout << "cpu->gpu time " << (time1-time0)/1000 << " ms" << endl;

			time0 = get_timestamp_dss();
			compute_TSDF(&scenes, &box_id, dataGPUmem,grid_size,encode_type,scale);
			time1 = get_timestamp_dss();
			cout << "compute time " << (time1-time0)/1000 << " ms" << endl;

			time0 = get_timestamp_dss();
			checkCUDA(__LINE__, cudaMemcpy(dataCPUmem, dataGPUmem,(len)*3*30*30*30*sizeof(float), cudaMemcpyDeviceToHost) );
			time1 = get_timestamp_dss();
			cout << "gpu->cpu time " << (time1-time0)/1000 << " ms" << endl;

			//clear for workaround
			scenes.clear();
			box_id.clear();

			//write TSDF to temp file because couldnt figure out how to pass it to Julia
			time0 = get_timestamp_dss();
			FILE * fid = fopen(tsdffile.c_str(),"wb");
			fwrite(dataCPUmem,sizeof(float),len*3*30*30*30,fid);
			fclose(fid);
			time1 = get_timestamp_dss();
			cout << "cpu->file " << (time1-time0)/1000 << " ms" << endl << endl;

			//Pass by reference paramater update
			count = len;
	
			//free memory
			delete scene;
			delete[] dataCPUmem;
			cudaFree(dataGPUmem);
			return 1;
		}
		return -1;
	}
}
