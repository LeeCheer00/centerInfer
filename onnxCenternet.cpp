#include <algorithm>
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

const std::string gSampleName = "TensorRT.sample_onnx_centernet";

using namespace nvinfer1;


static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int OUTPUT_CLS_SIZE = 2;
static const int OUTPUT_SIZE_H = 64;
static const int OUTPUT_SIZE_W = 64;

const std::string CLASSES[OUTPUT_CLS_SIZE]{"carringbag", "umbrella"};

const char* INPUT_BLOB_NAME0 = "data";
const char* OUTPUT_BLOB_NAME0 = "wh";
const char* OUTPUT_BLOB_NAME1 = "reg";
const char* OUTPUT_BLOB_NAME2 = "hm";
const char* OUTPUT_BLOB_NAME3 = "h_max";

samplesCommon::Args gArgs;

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& fileName, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(fileName, buffer, INPUT_H, INPUT_W);
}

bool onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    //Optional - uncomment below lines to view network layer information
    //config->setPrintLayerInfo(true);
    //parser->reportParsingInfo();

    if ( !parser->parseFromFile( locateFile(modelFile, gArgs.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        return false;
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);

    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }
    
    samplesCommon::enableDLA(builder, gArgs.useDLACore);
    
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    if (engine)
    {
        // serialize the engine, then close everything down
        trtModelStream = engine->serialize();
		// save the engine
        std::ofstream ofs("256M256.trt", std::ios::out | std::ios::binary);
        gLogInfo << "TRT engine saved: /home/ubuntu/SSD/software/TensorRT-5.1.2.2/bin.";
        ofs.write((char*)(trtModelStream ->data()), trtModelStream ->size());
        ofs.close();
        engine->destroy();
    }
    // engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

/*
void decodeHeatmap(float* wh, float* reg, float* hm, float* h_max, int batchSize, k=100) {
}
*/

void doInference(IExecutionContext& context, float* input, float* wh, float* reg, float* hm, float* h_max, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 5);
    void* buffers[5];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()

    int dataIndex = engine.getBindingIndex(INPUT_BLOB_NAME0),
    	whIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
    	regIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
    	hmIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME2),
    	h_maxIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME3);

    const int CLS_IP = 3;
    const int dataSize = batchSize * CLS_IP * INPUT_H * INPUT_W;
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

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/centernet/", "data/centernet/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    // Batch size
    // const int N = 10;
    // create a TensorRT model from the onnx model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};

    if (!onnxToTRTModel("centernet_mobilenetv2_10_objdet.onnx", 1, trtModelStream))
        gLogger.reportFail(sampleTest);

    assert(trtModelStream != nullptr);

    // read a random digit file
    srand(unsigned(time(nullptr)));
    uint8_t fileData[3 * INPUT_H * INPUT_W];
    int num = rand() % 10;
    readPGMFile(locateFile("test_data_set_0/input_0.pb", gArgs.dataDirs), fileData);


    float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0 - float(fileData[i] / 255.0);

    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // run inference
    const int outPutSize = 2 * OUTPUT_SIZE_H * OUTPUT_SIZE_W;
    float wh[outPutSize];
    float reg[outPutSize];
    float hm[outPutSize];
    float h_max[outPutSize];
    

    doInference(*context, data, wh, reg, hm, h_max, 1);

    // output the wh, reg, hm, h_max 
    int len = sizeof(wh) / sizeof(float);
    float *pWh = wh; 
    float *pReg = reg; 
    float *pHm = hm; 
    float *pH_max = h_max; 

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
    fs.close();


    // close the fileName
    fs.close();



    // postprocessing
    // decodeHeatmap(wh, reg, hm, h_max, dt, K);

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

}
