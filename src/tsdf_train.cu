//Include
#include "tsdf.h"

//Global variables
std::vector<Scene3D*> scenes;
std::vector<int> box_id;
int totalObjectCount = 0;
float scale = 100;
float context_pad =3;
std::vector<int> grid_size {3,30,30,30};
int encode_type =100;
int totalScenes = 0;
string file_list = "..//data//boxes_NYU_po_train_diff_nb2000_fb.list";
string data_root =  "..//data//";
string output_data = "..//data//julia_data//";

int main(int argc, char **argv){
	int requestedScene = atoi(argv[1]);
	

	FILE* fp = NULL;
	cout << "Loading file: " << file_list << endl << endl;
	fp = fopen(file_list.c_str(),"rb");
	if (fp==NULL) { cout << "Failed to open file: "<< file_list << endl; exit(EXIT_FAILURE); }


	while (feof(fp)==0)
	{
		Scene3D* scene = new Scene3D();
		unsigned int len = 0;
		fread((void*)(&len), sizeof(unsigned int), 1, fp);    
		if (len==0) return -1;
		scene->filename.resize(len);
		if (len>0) fread((void*)(scene->filename.data()), sizeof(char), len, fp);

		
		string s = scene->filename;
		scene->filename = data_root+scene->filename+".bin";

		fread((void*)(scene->R), sizeof(float), 9, fp);
		fread((void*)(scene->K), sizeof(float), 9, fp);
		fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);  
		fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 


		fread((void*)(&len),    sizeof(unsigned int),   1, fp);
		scene->objects.resize(len);
		if (len>0){
		  totalObjectCount += len;
		  for (int i=0; i<len; ++i){
		      Box3D box;
		      fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
		      fread((void*)(box.base),        sizeof(float), 9, fp);
		      fread((void*)(box.center),      sizeof(float), 3, fp);
		      fread((void*)(box.coeff),       sizeof(float), 3, fp);
		      //process box pad contex oreintation 
		      box = processbox (box, context_pad, grid_size[1]);
		      scene->objects[i]=box;
		      box_id.push_back(i);

			    uint8_t hasTarget = 0;
			    fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp);
			    if (hasTarget>0){
			      float box_tar_diff[6];
			      fread((void*)(box_tar_diff), sizeof(float), 6, fp);
			    }
		  }
		}
		scenes.push_back(scene);
		totalScenes++;

		if (totalScenes != requestedScene)
		{
			scenes.clear();
			box_id.clear();
			delete scene;
			continue;
		}

		cout << "Scene: " << totalScenes << " Boxes: " << len << " Bin: " << scene->filename << endl << endl;

		//Output files
		FILE* tempname = fopen("..//data//julia_data//temp.txt", "w");
		fprintf(tempname, "%s", s.substr(20).c_str());
		fclose(tempname);
		string tsdffile = output_data+"temp.tdsf";

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

		//write TSDF to temp file because couldnt figure out how to pass it to Julia
		time0 = get_timestamp_dss();
		FILE * fid = fopen(tsdffile.c_str(),"wb");
		fwrite(dataCPUmem,sizeof(float),len*3*30*30*30,fid);
		fclose(fid);
		time1 = get_timestamp_dss();
		cout << "cpu->file " << (time1-time0)/1000 << " ms" << endl << endl;

		//clear for workaround
		scenes.clear();
		box_id.clear();

		//free memory
		delete scene;
		delete[] dataCPUmem;
		cudaFree(dataGPUmem);
		
		//Dont calculate others
		break;
	}

	
	fclose(fp);
	return 0;
}
