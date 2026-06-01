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

__global__ void gammaKernel(
    float* img,
    int width,
    int height,
    int stepBytes,
    float gamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* row = (float*)((char*)img + y * stepBytes);
    int idx = x * 3;

    // Clamp first, then gamma — handles highlight magenta in one kernel
    float r = fminf(1.0f, fmaxf(0.0f, row[idx + 0]));
    float g = fminf(1.0f, fmaxf(0.0f, row[idx + 1]));
    float b = fminf(1.0f, fmaxf(0.0f, row[idx + 2]));

    row[idx + 0] = powf(r, gamma);
    row[idx + 1] = powf(g, gamma);
    row[idx + 2] = powf(b, gamma);
}

int main(int argc, char* argv[]) {

    LibRaw rawProcessor;

    int result = rawProcessor.open_file("DSC_9269.NEF");
    if(result != LIBRAW_SUCCESS) {
        std::cerr << "Failed to open RAW file!" << std::endl;
        return -1;
    }

    rawProcessor.imgdata.params.bright = 1.0f;
    rawProcessor.imgdata.params.highlight = 2;

    result = rawProcessor.unpack();
    if(result != LIBRAW_SUCCESS) {
        std::cerr << "Unpack failed!" << std::endl;
        return -1;
    }

    printf("=== RAW METADATA ===\n");
printf("raw_width=%d raw_height=%d\n", 
    rawProcessor.imgdata.sizes.raw_width,
    rawProcessor.imgdata.sizes.raw_height);
printf("width=%d height=%d\n", 
    rawProcessor.imgdata.sizes.width,
    rawProcessor.imgdata.sizes.height);
printf("left_margin=%d top_margin=%d\n",
    rawProcessor.imgdata.sizes.left_margin,
    rawProcessor.imgdata.sizes.top_margin);
printf("raw_pitch=%d\n",
    rawProcessor.imgdata.sizes.raw_pitch);
printf("black=%d maximum=%d\n",
    rawProcessor.imgdata.color.black,
    rawProcessor.imgdata.color.maximum);
printf("cam_mul: R=%.4f G=%.4f B=%.4f G2=%.4f\n",
    rawProcessor.imgdata.color.cam_mul[0],
    rawProcessor.imgdata.color.cam_mul[1],
    rawProcessor.imgdata.color.cam_mul[2],
    rawProcessor.imgdata.color.cam_mul[3]);
printf("filters=0x%08X\n",
    rawProcessor.imgdata.idata.filters);

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

    printf("srcStep(NPP)=%d  raw_pitch=%d  match=%s\n",
    srcStep,
    rawProcessor.imgdata.sizes.raw_pitch,  // print before recycle()!
    srcStep == rawProcessor.imgdata.sizes.raw_pitch ? "YES" : "NO");

    rawProcessor.recycle();

    // debayer on gpu
    NppiSize srcSize = { width, height };
    NppiRect srcROI  = { 0, 0, width, height };
    NppStreamContext streamCtx = {};  // zero-init = default stream

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

    printf("=== COLOR TWIST MATRIX ===\n");
printf("  %.4f %.4f %.4f  offset=%.4f\n", t00, t01, t02, offset0);
printf("  %.4f %.4f %.4f  offset=%.4f\n", t10, t11, t12, offset1);
printf("  %.4f %.4f %.4f  offset=%.4f\n", t20, t21, t22, offset2);
printf("r_gain=%.4f g_gain=%.4f b_gain=%.4f\n", r_gain, g_gain, b_gain);
printf("bl=%.4f wh=%.4f stretch=%.4f\n", bl, wh, stretch);

    nppiColorTwist_32f_C3R_Ctx(pDeviceRGB_32f, nStep32f, pDeviceRGB_32f, nStep32f, srcSize, aCombinedTwist,streamCtx);
    
    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    gammaKernel<<<blocks, threads>>>(
        pDeviceRGB_32f,
        width,
        height,
        nStep32f,
        0.4545f
    );

    nppiScale_32f8u_C3R_Ctx(pDeviceRGB_32f, nStep32f, d_rgbAligned, dstStep, srcSize, 0.0f, 1.0f, streamCtx);
    
    nppiFree(pDeviceRGB_32f);

    uchar3* h_rgb = (uchar3*)malloc(pixel_count * sizeof(uchar3));

    cudaMemcpy2D(
        h_rgb, width * sizeof(uchar3),
        d_rgbAligned, dstStep,
        width * sizeof(uchar3),
        height,
        cudaMemcpyDeviceToHost
    );

    nppiFree(d_rawAligned);
    nppiFree(d_rgbAligned);

    stbi_write_png("out.png", width, height, 3, (unsigned char*)h_rgb, width * sizeof(uchar3));

    free(h_rgb);

    return 0;
}