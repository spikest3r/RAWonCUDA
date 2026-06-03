#include <libraw/libraw.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <npp.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "helper_math.h"
#include <nvjpeg.h>
#include <fstream>
#include <nvtx3/nvtx3.hpp>
#include <tiffio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void launch_scale_16u32f_C3R_kernel(
    const unsigned short* pSrc, int srcStepBytes,
    float* pDst, int dstStepBytes,
    int width, int height,
    float nMin, float nMax, cudaStream_t stream);

void launch_strip_alpha_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width,
    int height, cudaStream_t stream);

void launch_c3_to_c4(const float* src, int srcPitch,
                          float* dst, int dstPitch,
                          int width, int height, cudaStream_t stream);

void launch_process_effects(float* image, int stepBytes, cudaSurfaceObject_t surface, int w, int h, float exposure, float contrast, float fH, float fS, float gamma, float saturation);