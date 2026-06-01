#define GLFW_INCLUDE_NONE 
#include "glad.h"
#include <GLFW/glfw3.h>
#include <libraw/libraw.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <npp.h>
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

float vertices[] = {
    // positions          // texture coords
     1.0f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
     1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
    -1.0f,  1.0f, 0.0f,   0.0f, 1.0f    // top left 
};
unsigned int indices[] = {  
    0, 1, 3, // first triangle
    1, 2, 3  // second triangle
};

const char* vs = R"(#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
})";

const char* fs = R"(#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

// The texture sampler uniform linked to texture unit 0
uniform sampler2D ourTexture; 

void main() {
    FragColor = texture(ourTexture, TexCoord);
})";

NppiBayerGridPosition get_bayer_type_fast(const LibRaw& rawProcessor) {
    unsigned int filters = rawProcessor.imgdata.idata.filters;
    if (filters < 1000) {
        throw std::runtime_error("Not implemented: non-CFA filter");
    }

    auto norm = [](int c) { return c == 3 ? 1 : c; };
    int c00 = norm((filters >> 0) & 3);
    int c01 = norm((filters >> 2) & 3);
    int c10 = norm((filters >> 4) & 3);
    int c11 = norm((filters >> 6) & 3);

    if (c00 == 0 && c01 == 1 && c10 == 1 && c11 == 2) return NPPI_BAYER_RGGB;
    if (c00 == 1 && c01 == 0 && c10 == 2 && c11 == 1) return NPPI_BAYER_GRBG;
    if (c00 == 1 && c01 == 2 && c10 == 0 && c11 == 1) return NPPI_BAYER_GBRG;
    if (c00 == 2 && c01 == 1 && c10 == 1 && c11 == 0) return NPPI_BAYER_BGGR;

    throw std::runtime_error("Unknown Bayer pattern: c00=" + std::to_string(c00) +
                             " c01=" + std::to_string(c01) +
                             " c10=" + std::to_string(c10) +
                             " c11=" + std::to_string(c11));
}

__global__ void process_effects(float* image, int stepBytes, cudaSurfaceObject_t surface, int w, int h, float exposure, float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= w || y >= h)
        return;

    float* row = (float*)((char*)image + y * stepBytes);
    int idx = x * 3;

    // Load values
    float r = row[idx + 0];
    float g = row[idx + 1];
    float b = row[idx + 2];
    
    // Exposure
    r *= exp2f(exposure);
    g *= exp2f(exposure);
    b *= exp2f(exposure);

    // Gamma Correction
    
    float gR = powf(fminf(1.0f, fmaxf(0.0f, r)), gamma);
    float gG = powf(fminf(1.0f, fmaxf(0.0f, g)), gamma);
    float gB = powf(fminf(1.0f, fmaxf(0.0f, b)), gamma);

    uchar4 out;
    out.x = __float2uint_rn(gR * 255.0f);
    out.y = __float2uint_rn(gG * 255.0f);
    out.z = __float2uint_rn(gB * 255.0f);
    out.w = 255;

    surf2Dwrite(out, surface, x * sizeof(uchar4), h - y - 1);
}

int main(int argc, char* argv[]) {
    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "NEF preview", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD v1\n";
        return -1;
    }

    LibRaw rawProcessor;

    int result = rawProcessor.open_file("DSC_9269.NEF");
    if(result != LIBRAW_SUCCESS) {
        std::cerr << "Failed to open RAW file!" << std::endl;
        glfwTerminate();
        return -1;
    }

    rawProcessor.imgdata.params.bright = 1.0f;
    rawProcessor.imgdata.params.highlight = 2;

    result = rawProcessor.unpack();
    if(result != LIBRAW_SUCCESS) {
        std::cerr << "Unpack failed!" << std::endl;
        glfwTerminate();
        return -1;
    }

    // get dimensions and prepare array
    int width = rawProcessor.imgdata.sizes.width;
    int height = rawProcessor.imgdata.sizes.height;
    
    int left_margin = rawProcessor.imgdata.sizes.left_margin;
    int top_margin = rawProcessor.imgdata.sizes.top_margin;

    int raw_stride = rawProcessor.imgdata.sizes.raw_width; // 6080, not raw_pitch/2
    unsigned short* pRealImage = rawProcessor.imgdata.rawdata.raw_image +
                                top_margin * raw_stride + left_margin;
                                 
    size_t pixel_count = (size_t)width * height;

    float r_gain = rawProcessor.imgdata.color.cam_mul[0] / rawProcessor.imgdata.color.cam_mul[1];
    float g_gain = 1.0f;
    float b_gain = rawProcessor.imgdata.color.cam_mul[2] / rawProcessor.imgdata.color.cam_mul[1];

    float m00 = rawProcessor.imgdata.color.rgb_cam[0][0], m01 = rawProcessor.imgdata.color.rgb_cam[0][1], m02 = rawProcessor.imgdata.color.rgb_cam[0][2];
    float m10 = rawProcessor.imgdata.color.rgb_cam[1][0], m11 = rawProcessor.imgdata.color.rgb_cam[1][1], m12 = rawProcessor.imgdata.color.rgb_cam[1][2];
    float m20 = rawProcessor.imgdata.color.rgb_cam[2][0], m21 = rawProcessor.imgdata.color.rgb_cam[2][1], m22 = rawProcessor.imgdata.color.rgb_cam[2][2];

    float norm = 1.0f / (float)rawProcessor.imgdata.color.maximum; // 1/4095

    float bl = (float)rawProcessor.imgdata.color.black * norm;  // 0.0
    float wh = 1.0f;                                             // 4095/4095
    float stretch = 1.0f / (wh - bl);                           // 1.0

    float r_scale = r_gain * stretch;
    float g_scale = g_gain * stretch;
    float b_scale = b_gain * stretch;

    // Pre-multiply Color Space Matrix by scaled White Balance
    float t00 = m00 * r_scale, t01 = m01 * g_scale, t02 = m02 * b_scale;
    float t10 = m10 * r_scale, t11 = m11 * g_scale, t12 = m12 * b_scale;
    float t20 = m20 * r_scale, t21 = m21 * g_scale, t22 = m22 * b_scale;

    // Bake Black Level Subtraction directly into the matrix offsets (4th column)
    float offset0 = -bl * (t00 + t01 + t02);
    float offset1 = -bl * (t10 + t11 + t12);
    float offset2 = -bl * (t20 + t21 + t22);

    Npp32f aCombinedTwist[3][4] = {
        { t00, t01, t02, offset0 },
        { t10, t11, t12, offset1 },
        { t20, t21, t22, offset2 }
    };

    auto debayerType = get_bayer_type_fast(rawProcessor);

    int srcStep, dstStep;

    Npp16u* d_rawAligned = nppiMalloc_16u_C1(width, height, &srcStep);
    Npp8u*  d_rgbAligned = nppiMalloc_8u_C3(width, height, &dstStep);

    cudaMemcpy2D(
        d_rawAligned, srcStep,
        pRealImage,   rawProcessor.imgdata.sizes.raw_width * sizeof(unsigned short),
        width * sizeof(Npp16u), height,
        cudaMemcpyHostToDevice
    );

    rawProcessor.recycle();

    // debayer on gpu
    NppiSize srcSize = { width, height };
    NppiRect srcROI  = { 0, 0, width, height };

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    NppStreamContext streamCtx = {};  // zero-init = default stream
    streamCtx.hStream = stream;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    Npp16u* d_rgb16 = nullptr;
    int rgb16Step;
    d_rgb16 = nppiMalloc_16u_C3(width, height, &rgb16Step);

    Npp32u shiftVal = 4;
    nppiLShiftC_16u_C1IR_Ctx(shiftVal, d_rawAligned, srcStep, srcSize, streamCtx);

    // debayer 16u -> 16u RGB
    NppStatus status = nppiCFAToRGB_16u_C1C3R_Ctx(
        d_rawAligned, srcStep,
        srcSize, srcROI,
        d_rgb16, rgb16Step,
        debayerType,
        NPPI_INTER_UNDEFINED,
        streamCtx
    );

    nppiFree(d_rawAligned);

    // scale 16u -> 8u
    nppiScale_16u8u_C3R_Ctx(
        d_rgb16, rgb16Step,
        d_rgbAligned, dstStep,
        srcSize,
        NPP_ALG_HINT_NONE,
        streamCtx
    );

    nppiFree(d_rgb16);

    Npp32f* pDeviceRGB_32f;
    int nStep32f;
    pDeviceRGB_32f = nppiMalloc_32f_C3(width, height, &nStep32f);

    nppiScale_8u32f_C3R_Ctx(d_rgbAligned, dstStep, pDeviceRGB_32f, nStep32f, srcSize, 0.0f, 1.0f, streamCtx);

    nppiFree(d_rgbAligned);

    nppiColorTwist_32f_C3R_Ctx(pDeviceRGB_32f, nStep32f, pDeviceRGB_32f, nStep32f, srcSize, aCombinedTwist,streamCtx);

    // get exec time
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "NPP Debayer and Post-Processing Execution Time: " << milliseconds << " ms" << std::endl;

    // clean up left over cuda
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // create gl quad for render and compile shader
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLuint shaderProgram;
    GLuint vertexShader, fragmentShader;

    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vs, NULL);
    glShaderSource(fragmentShader, 1, &fs, NULL);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // create gl texture, bind with cuda and copy data
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        nullptr
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    static cudaGraphicsResource* cudaResource;
    cudaArray_t cuArray;

    cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);

    dim3 threads(32, 32);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    process_effects<<<blocks, threads>>>(pDeviceRGB_32f, nStep32f, surface, width, height, 0.7f, 0.4545f);

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);

    auto textureRef = glGetUniformLocation(shaderProgram, "ourTexture");

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glActiveTexture(GL_TEXTURE0);

        glBindTexture(GL_TEXTURE_2D, texture);

        glUniform1i(textureRef, 0); 

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    nppiFree(pDeviceRGB_32f);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();

    return 0;
}