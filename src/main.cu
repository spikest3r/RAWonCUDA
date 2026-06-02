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
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "helper_math.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

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

__global__ void process_effects(float* image, int stepBytes, cudaSurfaceObject_t surface, int w, int h, float exposure, float contrast, float fH, float fS, float gamma, float saturation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= w || y >= h)
        return;

    float4* row = (float4*)((char*)image + y * stepBytes);
    float4 pixel = row[x];
    
    // Exposure
    pixel = pixel * exp2f(exposure);

    // Shadows & Highlights
    float L = pixel.x * 0.299f + pixel.y * 0.587f + pixel.z * 0.114f;

    float shadowMask    = 1.0f - smoothstep(0.0f, 0.75f, L);
    float highlightMask = smoothstep(0.25f, 1.0f, L);

    float scale = 1.0f + fS * shadowMask + fH * highlightMask;
    pixel = pixel * scale;

    // Contrast
    pixel = (pixel - 0.5f) * contrast + 0.5f;

    // Saturation
    pixel = L + (pixel - L) * saturation;

    // Gamma Correction
    float gR = powf(fminf(1.0f, fmaxf(0.0f, pixel.x)), gamma);
    float gG = powf(fminf(1.0f, fmaxf(0.0f, pixel.y)), gamma);
    float gB = powf(fminf(1.0f, fmaxf(0.0f, pixel.z)), gamma);

    // float to uchar
    uchar4 out;
    out.x = __float2uint_rn(gR * 255.0f);
    out.y = __float2uint_rn(gG * 255.0f);
    out.z = __float2uint_rn(gB * 255.0f);
    out.w = 255;

    surf2Dwrite(out, surface, x * sizeof(uchar4), h - y - 1);
}

__global__ void c3_to_c4(const float* src, int srcPitch,
                          float* dst, int dstPitch,
                          int width, int height)
{
    // each thread processes 4 pixels
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // src row as float4 pointer — 3 float4 reads = 4 pixels (12 floats)
    const float4* row_src = (const float4*)((const char*)src + y * srcPitch);
    float4*       row_dst = (float4*)      ((char*)      dst + y * dstPitch);

    // 3 coalesced 128-bit loads
    float4 c0 = row_src[(x * 3) / 4 + 0];  // R0 G0 B0 R1
    float4 c1 = row_src[(x * 3) / 4 + 1];  // G1 B1 R2 G2
    float4 c2 = row_src[(x * 3) / 4 + 2];  // B2 R3 G3 B3

    // unpack 4 pixels
    float4 p0 = { c0.x, c0.y, c0.z, 1.0f };
    float4 p1 = { c0.w, c1.x, c1.y, 1.0f };
    float4 p2 = { c1.z, c1.w, c2.x, 1.0f };
    float4 p3 = { c2.y, c2.z, c2.w, 1.0f };

    // 4 coalesced 128-bit stores
    row_dst[x + 0] = p0;
    row_dst[x + 1] = p1;
    row_dst[x + 2] = p2;
    row_dst[x + 3] = p3;
}

__global__ void scale_16u32f_C3R_kernel(
    const unsigned short* __restrict__ pSrc, int srcStepBytes,
    float* __restrict__ pDst, int dstStepBytes,
    int width, int height,
    float nMin, float nMax) 
{
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const unsigned short* srcRow = (const unsigned short*)((const char*)pSrc + y * srcStepBytes);
        float* dstRow = (float*)((char*)pDst + y * dstStepBytes);

        int srcIdx = x * 3;
        int dstIdx = x * 3;

        float scale = (nMax - nMin) / 65535.0f;

        dstRow[dstIdx + 0] = ((float)srcRow[srcIdx + 0] * scale) + nMin; // Red
        dstRow[dstIdx + 1] = ((float)srcRow[srcIdx + 1] * scale) + nMin; // Green
        dstRow[dstIdx + 2] = ((float)srcRow[srcIdx + 2] * scale) + nMin; // Blue
    }
}

int main(int argc, char* argv[]) {
    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(960, 640, "NEF preview", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD v1\n";
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

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

    int srcStep;

    Npp16u* d_rawAligned = nppiMalloc_16u_C1(width, height, &srcStep);
    
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

    Npp32f* pDeviceRGB_32f;
    int nStep32f;
    pDeviceRGB_32f = nppiMalloc_32f_C3(width, height, &nStep32f);

    dim3 threads(32, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    // more precision for float, better color range and image quality
    scale_16u32f_C3R_kernel<<<blocks, threads, 0, stream>>>(
        d_rgb16, rgb16Step, 
        pDeviceRGB_32f, nStep32f, 
        width, height, 
        0.0f, 1.0f
    );
    
    nppiColorTwist_32f_C3R_Ctx(pDeviceRGB_32f, nStep32f, pDeviceRGB_32f, nStep32f, srcSize, aCombinedTwist,streamCtx);

    Npp32f* pDeviceRGBA_32f;
    int nStepRGBA32f;
    pDeviceRGBA_32f = nppiMalloc_32f_C4(width, height, &nStepRGBA32f);

    dim3 grid1((width / 4 + threads.x - 1) / threads.x,
            (height     + threads.y - 1) / threads.y);

    c3_to_c4<<<grid1, threads, 0, stream>>>(pDeviceRGB_32f, nStep32f, pDeviceRGBA_32f, nStepRGBA32f, width, height);

    nppiFree(pDeviceRGB_32f);

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

    bool update = true; // first frame
    float exposure = 0.0f;
    float contrast = 1.0f;
    float factorH = 1.0f;
    float factorS = 1.0f;
    float saturation = 1.0f;

    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);

    auto textureRef = glGetUniformLocation(shaderProgram, "ourTexture");

    stbi_flip_vertically_on_write(1);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // compute frame if update flag is set
        if(update) {
            update = false;

            cudaGraphicsMapResources(1, &cudaResource, 0);
            cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray;

            cudaSurfaceObject_t surface = 0;
            cudaCreateSurfaceObject(&surface, &resDesc);

            process_effects<<<blocks, threads>>>(pDeviceRGBA_32f, nStepRGBA32f, surface, width, height, exposure, contrast, factorH, factorS, 0.4545f, saturation);

            cudaDestroySurfaceObject(surface);
            cudaGraphicsUnmapResources(1, &cudaResource, 0);
        }

        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Image");
            char buf[32];
            snprintf(buf, 32, "%f", exposure);
            if(ImGui::SliderFloat("Exposure", &exposure, -5.0f, 5.0f, buf)) {
                update = true;
            }
            snprintf(buf, 32, "%f", contrast);
            if(ImGui::SliderFloat("Contrast", &contrast, 0.0f, 2.0f, buf)) {
                update = true;
            }
            snprintf(buf, 32, "%f", factorH);
            if(ImGui::SliderFloat("Highlights", &factorH, 0.5f, 3.0f, buf)) {
                update = true;
            }
            snprintf(buf, 32, "%f", factorS);
            if(ImGui::SliderFloat("Shadows", &factorS, 0.5f, 3.0f, buf)) {
                update = true;
            }
            snprintf(buf, 32, "%f", saturation);
            if(ImGui::SliderFloat("Saturation", &saturation, -1.0f, 2.0f, buf)) {
                update = true;
            }
            if(ImGui::Button("Export to PNG")) {
                cudaGraphicsMapResources(1, &cudaResource, 0);

                cudaArray_t cuArray;
                cudaGraphicsSubResourceGetMappedArray(
                    &cuArray,
                    cudaResource,
                    0,
                    0
                );

                uchar4* hostBuffer = (uchar4*)malloc(width * height * sizeof(uchar4));

                size_t dstPitch = width * sizeof(uchar4);

                cudaMemcpy2DFromArray(
                    hostBuffer,               // dst host ptr
                    dstPitch,                 // dst pitch
                    cuArray,                  // src cudaArray
                    0, 0,                     // x,y offset in array
                    width * sizeof(uchar4),   // width in bytes
                    height,                   // height
                    cudaMemcpyDeviceToHost
                );

                cudaGraphicsUnmapResources(1, &cudaResource, 0);

                stbi_write_png("out.png", width, height, 4, (unsigned char*)hostBuffer, width * sizeof(uchar4));

                free(hostBuffer);
            }
            ImGui::End();
        }

        ImGui::Render();

        glUseProgram(shaderProgram);
        glActiveTexture(GL_TEXTURE0);

        glBindTexture(GL_TEXTURE_2D, texture);

        glUniform1i(textureRef, 0); 

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    nppiFree(pDeviceRGB_32f);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();

    return 0;
}