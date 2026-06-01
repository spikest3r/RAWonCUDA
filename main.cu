#include <libraw/libraw.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <npp.h>
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

    LibRaw rawProcessor;

    int result = rawProcessor.open_file("DSC_9269.NEF");
    if(result != LIBRAW_SUCCESS) {
        std::cerr << "Failed to open RAW file!" << std::endl;
        return -1;
    }

    result = rawProcessor.unpack();
    if(result != LIBRAW_SUCCESS) {
        std::cerr << "Unpack failed!" << std::endl;
        return -1;
    }

    // get dimensions and prepare array
    int width = rawProcessor.imgdata.sizes.raw_width;
    int height = rawProcessor.imgdata.sizes.raw_height;
    size_t pixel_count = (size_t)width * height;

    auto debayerType = get_bayer_type_fast(rawProcessor);

    unsigned short* h_raw = (unsigned short*)malloc(pixel_count * sizeof(unsigned short));
        
    memcpy(h_raw, 
            rawProcessor.imgdata.rawdata.raw_image, 
            pixel_count * sizeof(unsigned short));

    rawProcessor.recycle();

    int srcStep, dstStep;

    Npp16u* d_rawAligned = nppiMalloc_16u_C1(width, height, &srcStep);
    Npp8u*  d_rgbAligned = nppiMalloc_8u_C3(width, height, &dstStep);

    cudaMemcpy2D(
        d_rawAligned, srcStep,
        h_raw, width * sizeof(Npp16u),
        width * sizeof(Npp16u),
        height,
        cudaMemcpyHostToDevice
    );

    free(h_raw);

    // debayer on gpu
    NppiSize srcSize = { width, height };
    NppiRect srcROI  = { 0, 0, width, height };
    NppStreamContext streamCtx = {};  // zero-init = default stream

    Npp16u* d_rgb16 = nullptr;
    int rgb16Step;
    d_rgb16 = nppiMalloc_16u_C3(width, height, &rgb16Step);

    int shift = 2;

    Npp32u shiftVal = (Npp32u)shift;
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