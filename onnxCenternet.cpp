#include <algorithm>
#include <time.h>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const std::string gSampleName = "TensorRT.sample_onnx_centernet";

using namespace nvinfer1;


static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int INPUT_CLS_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 2;
static const int OUTPUT_SIZE_H = 64;
static const int OUTPUT_SIZE_W = 64;
static const int batchsize = 10;
static const int detsLength = 7;
static const int K = 100;
static const float CENTERNET_THRESH = 0.38;

const std::string CLASSES[OUTPUT_CLS_SIZE + 1]{"Background", "carringbag", "umbrella"}; // background offset
// const std::string img_list = "/home/ubuntu/SSD/software/TensorRT-5.1.2.2/data/centernet/carringBag_calibration.txt";
const std::string img_list = "/home/ubuntu/SSD/lq/tools/test_file.txt";

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME0 = "wh";
const char* OUTPUT_BLOB_NAME1 = "reg";
const char* OUTPUT_BLOB_NAME2 = "hm";
const char* OUTPUT_BLOB_NAME3 = "h_max";

const char* gNetworkName{nullptr};

samplesCommon::Args gArgs;

enum Objclassidx { Background, carringbag, umbrella };


std::vector<std::string> readTxt(std::string file)
{
  std::vector<std::string> file_vector;
  std::ifstream infile; 
  infile.open(file.data()); 
  assert(infile.is_open()); 

  std::string s;
  while(getline(infile,s))
  {
	  int nPos_1 = s.find(":");
	  file_vector.push_back(s.substr(0, nPos_1));
  }
  infile.close();
  return file_vector;
}

float* prepare_image(std::vector<string> infLst, int bs, std::vector<std::pair<int, int>>& WH){

	float *abuf = (float*)malloc(bs * INPUT_H * INPUT_W  * INPUT_CLS_SIZE * sizeof(float));

    int singleImageV = INPUT_H * INPUT_W * INPUT_CLS_SIZE;
    for(int j = 0; j < bs; j++)
    {	
        int  offset = singleImageV * j;
        std::string imageName = infLst[j];
        // std::cout << imageName << endl;
        
        cv::Mat image, im_resize, im_float;
        image = cv::imread(imageName);

        int imageW = image.cols;
        int imageH = image.rows;
        WH[j] = std::pair<int, int> (imageW, imageH);

        cv::resize(image, im_resize, cv::Size(INPUT_H, INPUT_W), cv::INTER_NEAREST);
        im_resize.convertTo(im_float, CV_32FC3);

        std::vector<cv::Mat> input_channels(INPUT_CLS_SIZE);
        for (int i = 0; i < 3; i++) {
            input_channels[i] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, &abuf[offset + i * INPUT_H * INPUT_W]);
        }
        
        cv::split(im_float, input_channels);
    }
    return abuf;
}

class ObjLocConf{ 
    public:
    	ObjLocConf(int batchid, int objIdx, int clsid, float conf, float xmin, float ymin, float xmax, float ymax);
    	void show() const;
        ObjLocConf():m_batchid(0), m_objIdx(0), m_clsid(0), m_conf(0), m_xmin(0), m_ymin(0), m_xmax(0), m_ymax(0){}

    public:
        int m_batchid;
        int m_objIdx;
        int m_clsid;
        float m_conf;
        float m_xmin;
        float m_ymin;
        float m_xmax;
        float m_ymax;
    	
};

ObjLocConf::ObjLocConf(int batchid, int objIdx, int clsid, float conf, float xmin, float ymin, float xmax, float ymax): m_batchid(batchid), m_objIdx(objIdx), m_clsid(clsid), m_conf(conf), m_xmin(xmin), m_ymin(ymin), m_xmax(xmax), m_ymax(ymax) {};
void ObjLocConf::show() const{
    gLogInfo << "\t\t第"<< m_batchid <<"张: 第" << m_objIdx << "个检测目标，类别: "<< CLASSES[m_clsid] << "，置信度为:" << m_conf << "，坐标为:" << "(" << m_xmin << "," << m_ymax << "), (" << m_xmax << "," << m_ymin <<")" << endl;
    // printf("%s类置信度%f,坐标位置为:(%f, %f), (%f, %f).\n", CLASSES[m_clsid], m_conf, m_xmin, m_ymax, m_xmax, m_ymin);
}

// cpu nms
bool nms(std::vector<ObjLocConf> &objs, float nms_threshold){
      
    auto cal_area = [](const ObjLocConf &objc) -> float { return (objc.m_ymax-objc.m_ymin)*(objc.m_xmax-objc.m_xmin);};

    auto overlap2D =[](ObjLocConf &obj1, ObjLocConf &obj2) -> float { 
        auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
            if (x1min > x2min)  {
                std::swap(x1min, x2min);
                std::swap(x1max, x2max);
            }
            return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
        };

        float overlapX = overlap1D(obj1.m_xmin, obj1.m_xmax, obj2.m_xmin, obj2.m_xmax);
        float overlapY = overlap1D(obj1.m_ymin, obj1.m_ymax, obj2.m_ymin, obj2.m_ymax);
        return overlapX * overlapY;
    };
     
    std::stable_sort(objs.begin(), objs.end(), [](const ObjLocConf &b1, const ObjLocConf &b2) -> bool { return b1.m_conf > b2.m_conf;});

    int nums = objs.size();
        
    vector<float> area(nums);
    for (int oidx = 0; oidx < nums; oidx++) {   
        area[oidx] = cal_area(objs[oidx]);
    }

    vector<bool> keep(nums, true);
    vector<class ObjLocConf> keep_objs;

    for (int oidx = 0; oidx < nums; oidx++){
            
        if (keep[oidx] == false){   continue;}

        int c_classidx = objs[oidx].m_clsid;
            
        for (int aidx = oidx + 1; aidx < nums; aidx++) {

            if (keep[aidx] == false){ continue;}
                
            int a_classidx = objs[aidx].m_clsid;                
                
            float inter_area = overlap2D(objs[aidx], objs[oidx]);
                
            if (inter_area > 0){
                float overlap2 = (float)inter_area / (area[oidx] + area[aidx] - inter_area);

                if ((c_classidx == Objclassidx::carringbag && a_classidx == Objclassidx::umbrella) ||
                    (c_classidx == Objclassidx::umbrella && a_classidx == Objclassidx::carringbag) ){
                    if (overlap2 > nms_threshold){  
                        keep[aidx] = false;
                    }
                }
            }
        }
    }

    std::vector<int> rcder;
    for (int oidx = 0; oidx < nums; oidx++) {
        if (keep[oidx]) {
            rcder.push_back(oidx);
        }
    }
    int keep_num = rcder.size();
    if (keep_num == 0) {
        return false;
    }

    for (int kidx = 0; kidx < keep_num; kidx++) {
        keep_objs.push_back(objs[rcder[kidx]]);
    }

    objs = keep_objs;
    return true;
}


// argsort
template<typename T> std::vector<int> argsort(const std::vector<T>& array)
{
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

	return array_index;
}

// ctdet decode
std::vector<float> decode(int bs, float* wh, float* reg, float* hm, float* h_max, int K) {
    const int stride = OUTPUT_SIZE_H * OUTPUT_SIZE_W;
    // const int detsSize = K * bs * detsLength;
    std::vector<float> dets;

    int singleImageSize = OUTPUT_SIZE_H * OUTPUT_SIZE_W * OUTPUT_CLS_SIZE;
    for (int bsidx = 0; bsidx < bs; bsidx++){ 
        float *b_hm = (float*)malloc(singleImageSize * sizeof(float)); 
        float *b_hmax = (float*)malloc(singleImageSize * sizeof(float)); 
        float *b_wh = (float*)malloc(singleImageSize * sizeof(float)); 
        float *b_reg = (float*)malloc(singleImageSize * sizeof(float)); 
        int offset = bsidx * singleImageSize;
        std::vector<float> b_score(singleImageSize);

        // b_hm, b_hmax, b_wh, b_reg, b_score 赋值
        for (int idx = 0; idx < singleImageSize; idx++) { 
            b_wh[idx] = wh[offset + idx];
            b_reg[idx] = reg[offset + idx];
            b_hm[idx] = hm[offset + idx];
            b_hmax[idx] = h_max[offset + idx];

            if (b_hm[idx] == b_hmax[idx]){
                b_score[idx] = b_hmax[idx];
            }else{
                b_score[idx] = 0;
            } 
        }
            // typedef std::vector<int> vt_type;
            // vt_type topk_ind_r;
        std::vector<int> topk_ind = argsort(b_score); 
        int topk_len = topk_ind.size();
        
            // topk_ind_r = argsort(b_score); 
            // for (vt_type::const_reverse_iterator topk_ind = topk_ind_r.rbegin(), isize = topk_ind_r.rend(); topk_ind != isize; ++topk_ind);

            // batchsize * K * 7 processing
        for (int k = 0; k < K; k++) { 
            int real_idx = topk_len - k - 1;
            int cls_id = topk_ind[real_idx] / stride;
            int center_index = topk_ind[real_idx] % stride; 
            int xs = center_index % OUTPUT_SIZE_W;
            int ys = center_index / OUTPUT_SIZE_W;

            float reg_x = b_reg[center_index];
            float reg_y = b_reg[center_index + stride];

            float w_half = b_wh[center_index] * 0.5; 
            float h_half = b_wh[center_index + stride] * 0.5;


            dets.push_back(float(bsidx));
            dets.push_back(float(cls_id));
            dets.push_back(b_score[topk_ind[real_idx]]);
            dets.push_back(xs + reg_x - w_half);
            dets.push_back(ys + reg_y - h_half);
            dets.push_back(xs + reg_x + w_half);
            dets.push_back(ys + reg_y + h_half);


        }
        free(b_hm);
        free(b_hmax);
        free(b_wh);
        free(b_reg);
    }
	return dets;

}


float* mat_to_array(cv::Mat in, int bs){
    int arrayCount = 0;
    int channel = INPUT_CLS_SIZE;
    int height = INPUT_H;
    int width = INPUT_W;
    float *out = (float*)malloc(batchsize * INPUT_H * INPUT_W* 3 * sizeof(float)); 
    for (int bsCount = 0; bsCount < bs; bsCount++){
        for (int c = 0; c < channel; c++) {
            for (int i=0; i < height; i++) {
                 for (int j =0; j < width; j++,arrayCount++){
                    out[arrayCount] = in.at<cv::Vec3f>(i,j)[c];
                    //printf("数组第%d个：第%d行，第%d列，第%d通道的值：%d\n",arrayCount, i+1, j+1, c+1, int(out[arrayCount]));
                }
            }
        }
    }
    //printf("everything is down.\nlength of the array is:%d.\n", arrayCount);
    return out;
}

// output the wh, reg, hm, h_max 
void writefeature(float*wh, float* reg, float* hm, float* h_max, float* data, int bs){ 
    int len = bs * OUTPUT_SIZE_W * OUTPUT_SIZE_H * OUTPUT_CLS_SIZE; 
    int outlen = bs * 3 * INPUT_H * INPUT_W;
    float *pWh = wh; 
    float *pReg = reg; 
    float *pHm = hm; 
    float *pH_max = h_max; 
    float *pData = data;

    fstream fs;

    fs.open("output.txt", ios::out);

    // printf("wh:\n");
    fs.write("wh:\n", 4);
    for (int i = 0; i < len; i++) {
        fs << *pWh++;
        fs.write("\n",1);
    }

    // reg output:
    fs.write("reg:\n", 5);
    for (int i = 0; i < len; i++) { 
        fs << *pReg++;
        fs.write("\n",1);
    }
    fs.write("\n",1);

    // hm output: 
    fs.write("hm:\n", 4);
    for (int i = 0; i < len; i++) { 
        fs << *pHm++;
        fs.write("\n",1);
    }
    fs.write("\n",1);

    // h_max output:
    fs.write("h_max:\n", 7);
    for (int i = 0; i < len; i++) { 
        // printf("%f", *p++);
        fs << *pH_max++;
        fs.write("\n",1);
    }
    fs.write("\n",1);

    fs.write("data:\n", 6);
    for (int i = 0; i < outlen; i++) {
        fs << *pData++;
        fs.write("\n", 1);
    }

    // close the fileName
    fs.close();
    gLogInfo << "feature map saved in output.txt.";
}


void doInference(IExecutionContext& context, float* input, float* wh, float* reg, float* hm, float* h_max, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 5);
    void* buffers[5];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()

    int dataIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
    	whIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
    	regIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
    	hmIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME2),
    	h_maxIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME3);
    /* 
    Dims3 inputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(INPUT_BLOB_NAME)));
    Dims3 whDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(OUTPUT_BLOB_NAME0)));
    printf("data:%d, %d, %d, %d\n", inputDims.d[0], inputDims.d[1], inputDims.d[2], inputDims.d[3]);
    printf("wh:%d, %d, %d, %d\n", whDims.d[0], whDims.d[1], whDims.d[2], whDims.d[3]);
    */

    const int CLS_IP = 3;
    const int dataSize = batchSize * CLS_IP * INPUT_H * INPUT_W;
    // printf("input size is:%d\n", dataSize);
    const int CLS_OP = 2;
    const int commonSize = batchSize * CLS_OP * OUTPUT_SIZE_H * OUTPUT_SIZE_W;
    const int whSize = commonSize;
    const int regSize = commonSize;
    const int hmSize = commonSize; 
    const int h_maxSize = commonSize;

    int inputIndex{};
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
            inputIndex = b;
    }

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], dataSize * sizeof(float))); // data

    for (int tempBuffIndex = 1; tempBuffIndex < engine.getNbBindings(); tempBuffIndex++) {
        CHECK(cudaMalloc(&buffers[tempBuffIndex], commonSize * sizeof(float))); // wh, reg, hm, h_max
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, dataSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);

    for (int tempBuffIndex = 1; tempBuffIndex < engine.getNbBindings(); tempBuffIndex++) {
		if (tempBuffIndex == whIndex)
        	CHECK(cudaMemcpyAsync(wh, buffers[tempBuffIndex], whSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
		else if (tempBuffIndex == regIndex)
        	CHECK(cudaMemcpyAsync(reg, buffers[tempBuffIndex], regSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
		else if (tempBuffIndex == hmIndex)
        	CHECK(cudaMemcpyAsync(hm, buffers[tempBuffIndex], hmSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        else if (tempBuffIndex == h_maxIndex) 
            CHECK(cudaMemcpyAsync(h_max, buffers[tempBuffIndex], h_maxSize *  sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[dataIndex]));
    CHECK(cudaFree(buffers[whIndex]));
    CHECK(cudaFree(buffers[regIndex]));
    CHECK(cudaFree(buffers[hmIndex]));
    CHECK(cudaFree(buffers[h_maxIndex]));
}


int main(int argc, char** argv) {

    // load the local file engine
    // std::string engine_file = "/home/ubuntu/SSD/software/TensorRT-5.1.2.2/bin/fp16-batch10.engine";
    std::string engine_file = "/home/ubuntu/SSD/software/TensorRT-5.1.2.2/bin/int8-onnx-centernet.engine";
    std::ifstream in_file(engine_file.c_str(), std::ios::in | std::ios::binary);
    if (!in_file.is_open()) {
        fprintf(stderr, "fail to open file to write: %s\n", engine_file.c_str());
    }
    std::streampos begin, end;
    begin = in_file.tellg();
    in_file.seekg(0, std::ios::end); 
    end = in_file.tellg();
    std::size_t size = end - begin;
    fprintf(stdout, "engine file size: %lu bytes\n", size);
    in_file.seekg(0, std::ios::beg);
    std::unique_ptr<unsigned char[]> engine_data(new unsigned char[size]);
    in_file.read((char*)engine_data.get(), size);
    in_file.close(); 

    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine((const void*)engine_data.get(), size, nullptr);
    assert(engine != nullptr);
    // trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // run inference
    const int outPutSize = batchsize * OUTPUT_SIZE_H * OUTPUT_SIZE_W * OUTPUT_CLS_SIZE; 
    float wh[outPutSize];
    float reg[outPutSize];
    float hm[outPutSize];
    float h_max[outPutSize];


    // double dur = 0;
    // double totalDur = 0;
    clock_t start, finish;
    start = clock(); 
    float totalDur = 0;
	std::vector<std::string> infLst = readTxt(img_list);
    gLogInfo << "Infering...";
    for (int btNm = 0; btNm < int (infLst.size() / batchsize); btNm++){ // todo :when batch is incompelte.
        std::vector<string> fInBc(infLst.begin() + btNm * batchsize, infLst.begin() + btNm * batchsize + batchsize);
		std::vector<std::pair<int, int>> WH(batchsize); 

		float *blob = prepare_image(fInBc, batchsize, WH);
        // start = clock();
    	doInference(*context, blob, wh, reg, hm, h_max, batchsize); 
        // printf("use Time:%f, %d, %f\n", totalDur/CLOCKS_PER_SEC, batchsize, totalDur/CLOCKS_PER_SEC/batchsize);

        // writefeature(wh, reg, hm, h_max, data, batchsize);
        // gLogInfo << "第" << btNm<< "个batch:\n";

        std::vector<float> dets = decode(batchsize, wh, reg, hm, h_max, K);
        // finish = clock();
        // dur = double (finish - start);
		// totalDur += dur;

        for (int indInBt = 0; indInBt < batchsize; ++indInBt) {
            // every batch's top K index.
            std::vector<int> indexInBatches;
            for (int j = 0; j < K; ++j) {
                    indexInBatches.push_back(indInBt * K + j);
            } 

            std::vector<int> top_indices;
            for (int k = 0; k < K; ++k){
                if(dets[indexInBatches[k] * detsLength + 2] > CENTERNET_THRESH){
                    top_indices.push_back(indexInBatches[k]);
                }else{
                    continue;
                } 
            }
            int num_objs = top_indices.size();

            float imageW = std::get<0>(WH[indInBt]);
            float imageH = std::get<1>(WH[indInBt]);

            float scale_w = float(imageW) / INPUT_W;
            float scale_h = float(imageH) / INPUT_H; 

            std::vector<class ObjLocConf> objs(num_objs);

            for (int j = 0; j < num_objs; j++) {
                int clsid = dets[top_indices[j] * detsLength + 1] + 1;
                float conf = dets[top_indices[j] * detsLength + 2];
                float xmin = dets[top_indices[j] * detsLength + 3] * scale_w;
                float ymin = dets[top_indices[j] * detsLength + 4] * scale_h;
                float xmax = dets[top_indices[j] * detsLength + 5] * scale_w;
                float ymax = dets[top_indices[j] * detsLength + 4] * scale_h;

                xmin = std::max(0.0f, xmin);
                ymin = std::max(0.0f, ymin);
                xmax = std::min(xmax, imageW - 1);
                ymin = std::max(ymax, imageH - 1); 


                objs[j] = ObjLocConf(indInBt, j, clsid, conf, xmin, ymin, xmax, ymax);
            }
            bool res = nms(objs, 0.5);
            int imageIndex = btNm * batchsize + indInBt;
            gLogInfo << infLst[imageIndex] << ":";
            if (res == true) {
                int rmOjbsNum = objs.size();
                int flagC = 0;
                int flagU = 0;
                for (int i = 0; i < rmOjbsNum; i++) {
                    if (objs[i].m_clsid == 1){
                        gLogInfo << "carringbag,";
                    	flagC = 1;
                    }
                    else if (objs[i].m_clsid == 2){
                        gLogInfo << "umbrella,";
                    	flagU = 1;
                    }
                }
                if (flagC && flagU) gLogInfo << "both detected.";

            }else{
                gLogInfo << "nothing detected.";
            }
            gLogInfo << "\n";
            // gLogInfo << "\t第" << indInBt << "张图:\n";
            // if (res == true) {
            //     int rmOjsNum = objs.size();

            //     for (int i = 0; i < rmOjsNum; i++) {
            //         gLogInfo << "\t\tNMS后:第" << i << "个:";
            //         objs[i].show();
            //     }

            // }else {
            //     gLogInfo << "\t\t里没检测到手拎包、雨伞目标\n";
            // }
        }
        // finish = clock();
        // float Dur = finish - start;
        // totalDur += Dur;
    }

    finish = clock();
    totalDur = finish - start;
    // printf("use Time:%f, %d, %f\n", totalDur/CLOCKS_PER_SEC, infLst.size(), totalDur/CLOCKS_PER_SEC/infLst.size());


    // printf("use Time:%f, %d, %f\n", totalDur/CLOCKS_PER_SEC, 590, totalDur/CLOCKS_PER_SEC/590);
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
