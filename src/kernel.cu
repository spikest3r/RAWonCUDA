#include "kernel.cuh"

// kernels

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

__global__ void strip_alpha_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int pixel_idx = y * width + x;

        const uchar4* src_pixels = (const uchar4*)src;
        
        int dst_byte_idx = ((height - y) * width + x) * 3; // flip since GL textures are flipped

        uchar4 p = src_pixels[pixel_idx];

        dst[dst_byte_idx + 0] = p.x; // R
        dst[dst_byte_idx + 1] = p.y; // G
        dst[dst_byte_idx + 2] = p.z; // B
    }
}

// helpers

void launch_scale_16u32f_C3R_kernel(
    const unsigned short* pSrc, int srcStepBytes,
    float* pDst, int dstStepBytes,
    int width, int height,
    float nMin, float nMax, cudaStream_t stream) 
{
    dim3 threads(32, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    scale_16u32f_C3R_kernel<<<blocks, threads, 0, stream>>>(
        pSrc, srcStepBytes, 
        pDst, dstStepBytes, 
        width, height, 
        nMin, nMax
    );
}

void launch_c3_to_c4(const float* src, int srcPitch,
                          float* dst, int dstPitch,
                          int width, int height, cudaStream_t stream)
{
    dim3 threads(32, 16);
    dim3 grid1((width / 4 + threads.x - 1) / threads.x,
            (height     + threads.y - 1) / threads.y);
    c3_to_c4<<<grid1, threads, 0, stream>>>(src, srcPitch, dst, dstPitch, width, height);
}

void launch_process_effects(float* image, int stepBytes, cudaSurfaceObject_t surface, int w, int h, float exposure, float contrast, float fH, float fS, float gamma, float saturation) {
    dim3 threads(32, 16);
    dim3 blocks(
        (w + threads.x - 1) / threads.x,
        (h + threads.y - 1) / threads.y
    );

    process_effects<<<blocks, threads>>>(image, stepBytes, surface, w, h, exposure, contrast, fH, fS, gamma, saturation);
}

void launch_strip_alpha_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width,
    int height, cudaStream_t stream) 
{
    dim3 threads(32, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    strip_alpha_kernel<<<blocks, threads, 0, stream>>>(src, dst, width, height);
}