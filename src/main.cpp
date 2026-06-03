#define GLFW_INCLUDE_NONE 
#include "glad.h"
#include <GLFW/glfw3.h>
#include "kernel.cuh"

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

#include <tiffio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TILE_WIDTH  256
#define TILE_HEIGHT 256

int write_8bit_tiff_tiled(const char *filename,
                           const unsigned char *pixels,
                           uint32_t width,
                           uint32_t height)
{
    TIFF *tif = TIFFOpen(filename, "w");
    if (!tif) return -1;

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   8);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,       COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_ZIPQUALITY,        6);
    TIFFSetField(tif, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_TILEWIDTH,       TILE_WIDTH);
    TIFFSetField(tif, TIFFTAG_TILELENGTH,      TILE_HEIGHT);
    uint16_t extra = EXTRASAMPLE_UNASSALPHA;
    TIFFSetField(tif, TIFFTAG_EXTRASAMPLES,   1, &extra);

    tmsize_t tile_size = TIFFTileSize(tif);
    unsigned char *tile_buf = (unsigned char *)malloc(tile_size);
    if (!tile_buf) { TIFFClose(tif); return -1; }

    for (uint32_t ty = 0; ty < height; ty += TILE_HEIGHT) {
        for (uint32_t tx = 0; tx < width; tx += TILE_WIDTH) {

            memset(tile_buf, 0, tile_size);

            for (uint32_t row = 0; row < TILE_HEIGHT; row++) {
                uint32_t img_y = (height - 1) - (ty + row);  /* flip: read from bottom up */
                if (img_y >= height) break;

                uint32_t copy_w = TILE_WIDTH;
                if (tx + copy_w > width) copy_w = width - tx;

                memcpy(tile_buf + row * TILE_WIDTH * 4,
                    pixels + (img_y * width + tx) * 4,
                    copy_w * 4);
            }

            if (TIFFWriteTile(tif, tile_buf, tx, ty, 0, 0) < 0) {
                free(tile_buf);
                TIFFClose(tif);
                return -1;
            }
        }
    }

    free(tile_buf);
    TIFFClose(tif);
    return 0;
}

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

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cout << "Usage: " << "cudaraw [image.raw]" << std::endl;
        return -1;
    }

    nvtxRangePush("Init GLFW");
    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(960, 640, "RAW preview", NULL, NULL);
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
    nvtxRangePop();

    nvtxRangePush("LibRaw");
    LibRaw rawProcessor;

    int result = rawProcessor.open_file(argv[1]);
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
    nvtxRangePop();

    nvtxRangePush("debayer_init");

    int srcStep;

    Npp16u* d_rawAligned = nppiMalloc_16u_C1(width, height, &srcStep);
    
    cudaMemcpy2D(
        d_rawAligned, srcStep,
        pRealImage,   rawProcessor.imgdata.sizes.raw_width * sizeof(unsigned short),
        width * sizeof(Npp16u), height,
        cudaMemcpyHostToDevice
    );

    int whiteLevel = rawProcessor.imgdata.color.maximum;
    int depth = (int)floor(log2(whiteLevel + 1)); 

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

    Npp32u shiftVal = (Npp32u)(16 - depth);
    nppiLShiftC_16u_C1IR_Ctx(shiftVal, d_rawAligned, srcStep, srcSize, streamCtx);

    nvtxRangePush("debayer");
    // debayer 16u -> 16u RGB
    NppStatus status = nppiCFAToRGB_16u_C1C3R_Ctx(
        d_rawAligned, srcStep,
        srcSize, srcROI,
        d_rgb16, rgb16Step,
        debayerType,
        NPPI_INTER_UNDEFINED,
        streamCtx
    );
    nvtxRangePop();

    nppiFree(d_rawAligned);

    Npp32f* pDeviceRGB_32f;
    int nStep32f;
    pDeviceRGB_32f = nppiMalloc_32f_C3(width, height, &nStep32f);

    // more precision for float, better color range and image quality
    launch_scale_16u32f_C3R_kernel(
        d_rgb16, rgb16Step, 
        pDeviceRGB_32f, nStep32f, 
        width, height, 
        0.0f, 1.0f, stream
    );
    
    nvtxRangePush("color_twist");
    nppiColorTwist_32f_C3R_Ctx(pDeviceRGB_32f, nStep32f, pDeviceRGB_32f, nStep32f, srcSize, aCombinedTwist,streamCtx);
    nvtxRangePop();

    Npp32f* pDeviceRGBA_32f;
    int nStepRGBA32f;
    pDeviceRGBA_32f = nppiMalloc_32f_C4(width, height, &nStepRGBA32f);

    launch_c3_to_c4(pDeviceRGB_32f, nStep32f, pDeviceRGBA_32f, nStepRGBA32f, width, height, stream);

    nppiFree(pDeviceRGB_32f);
    nvtxRangePop();

    // get exec time
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "NPP Debayer and Post-Processing Execution Time: " << milliseconds << " ms" << std::endl;

    // clean up left over cuda
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

    // initialize nvjpeg
    nvjpegHandle_t nvjpeg_handle;
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encoder_params;

    nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL, &nvjpeg_handle);
    nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, stream);
    nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, stream);

    nvjpegEncoderParamsSetQuality(encoder_params, 90, stream);

    nvjpegEncoderParamsSetSamplingFactors(encoder_params, NVJPEG_CSS_444, stream);

    // variables for ui
    bool update = true; // first frame
    float exposure = 0.0f;
    float contrast = 1.0f;
    float factorH = 1.0f;
    float factorS = 1.0f;
    float saturation = 1.0f;
    bool applyEffects = true;

    // pre loop preparation
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    auto textureRef = glGetUniformLocation(shaderProgram, "ourTexture");

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

            if(applyEffects) {
                launch_process_effects(pDeviceRGBA_32f, nStepRGBA32f, surface, width, height, exposure, contrast, factorH, factorS, 0.4545f, saturation);
            } else {
                launch_process_effects(pDeviceRGBA_32f, nStepRGBA32f, surface, width, height, 0.0f, 1.0f, 1.0f, 1.0f, 0.4545f, 1.0f);
            }

            cudaDestroySurfaceObject(surface);
            cudaGraphicsUnmapResources(1, &cudaResource, 0);
        }

        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Image");
            if(ImGui::Button("Reset to Default")) {
                exposure = 0.0f;
                contrast = 1.0f;
                factorH = 1.0f;
                factorS = 1.0f;
                saturation = 1.0f;
                update = true;
            }
            if(ImGui::Button("Toggle effects")) {
                applyEffects = !applyEffects;
                update = true;
            }
            ImGui::SameLine();
            ImGui::Text(applyEffects ? "YES" : "NO");
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
            if(ImGui::Button("Export to JPEG (CUDA)")) {
                unsigned char* d_linearRGB4 = nullptr;
                unsigned char* d_linearRGB3 = nullptr;

                nvtxRangePush("jpeg_encode_init");
                cudaMallocAsync((void**)&d_linearRGB4, width * height * 4, stream);

                cudaGraphicsMapResources(1, &cudaResource, stream);
                cudaArray_t cuArray;
                cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

                cudaMemcpy2DFromArrayAsync(
                    d_linearRGB4,
                    width * 4,
                    cuArray,
                    0, 0,
                    width * 4,
                    height,
                    cudaMemcpyDeviceToDevice,
                    stream
                );

                cudaGraphicsUnmapResources(1, &cudaResource, stream);

                cudaMallocAsync((void**)&d_linearRGB3, width * height * 3, stream);

                launch_strip_alpha_kernel(d_linearRGB4, d_linearRGB3, width, height, stream);

                cudaFreeAsync(d_linearRGB4, stream);

                nvjpegImage_t img_to_encode;
                img_to_encode.channel[0] = d_linearRGB3;
                img_to_encode.channel[1] = nullptr;
                img_to_encode.channel[2] = nullptr;
                img_to_encode.channel[3] = nullptr;
                img_to_encode.pitch[0] = width * 3;
                img_to_encode.pitch[1] = 0;
                img_to_encode.pitch[2] = 0;
                img_to_encode.pitch[3] = 0;

                size_t max_stream_length = 0;
                nvjpegEncodeGetBufferSize(nvjpeg_handle, encoder_params, width, height, &max_stream_length);

                nvjpegStatus_t status;

                nvtxRangePush("nvjpegEncodeImage");
                status = nvjpegEncodeImage(nvjpeg_handle, encoder_state, encoder_params,
                                        &img_to_encode, NVJPEG_INPUT_RGBI, width, height, stream);
                if (status != NVJPEG_STATUS_SUCCESS) {
                    printf("nvjpegEncodeImage failed with status: %d\n", status);
                }
                nvtxRangePop();

                size_t jpeg_size = 0;
                nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL, &jpeg_size, stream); // <-- Correct Name
                
                cudaStreamSynchronize(stream);

                unsigned char* h_jpegBuffer = nullptr;
                cudaHostAlloc((void**)&h_jpegBuffer, jpeg_size, cudaHostAllocDefault);

                nvtxRangePush("nvjpegEncodeRetrieveBitstream");
                nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, h_jpegBuffer, &jpeg_size, stream);
                nvtxRangePop();

                cudaStreamSynchronize(stream);
                cudaFreeAsync(d_linearRGB3, stream);

                nvtxRangePush("Write to disk");

                std::ofstream outFile("output.jpg", std::ios::out | std::ios::binary);
                if (outFile.is_open()) {
                    outFile.write((char*)h_jpegBuffer, jpeg_size);
                    outFile.close();
                }

                nvtxRangePop();

                cudaFreeHost(h_jpegBuffer);
                nvtxRangePop();
            }
            if(ImGui::Button("Export to TIFF (CPU)")) {
                unsigned char* d_linearRGB4 = nullptr;

                nvtxRangePush("encode_h2d");
                cudaMalloc((void**)&d_linearRGB4, width * height * 4);

                cudaGraphicsMapResources(1, &cudaResource, 0);
                cudaArray_t cuArray;
                cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

                cudaMemcpy2DFromArrayAsync(
                    d_linearRGB4,
                    width * 4,
                    cuArray,
                    0, 0,
                    width * 4,
                    height,
                    cudaMemcpyDeviceToDevice,
                    stream
                );

                cudaGraphicsUnmapResources(1, &cudaResource, 0);

                unsigned char* h_linearRGB4 = nullptr;
                cudaHostAlloc((void**)&h_linearRGB4, width * height * 4, cudaHostAllocDefault);

                cudaMemcpy(h_linearRGB4, d_linearRGB4, width * height * 4, cudaMemcpyDeviceToHost);

                cudaFree(d_linearRGB4);

                nvtxRangePush("TIFF on CPU");
                write_8bit_tiff_tiled("output.tiff", h_linearRGB4, width, height);
                nvtxRangePop();

                cudaFreeHost(h_linearRGB4);
                nvtxRangePop();
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

    cudaStreamDestroy(stream);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();

    return 0;
}