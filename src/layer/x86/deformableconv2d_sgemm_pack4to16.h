// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "convolution_sgemm_pack4to16.h"

static void deformableconv2d_im2col_sgemm_pack4to16_avx512(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, const Option& opt)
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);
    const bool offset_not_pack = offset.elempack == 1;
    const bool mask_not_pack = has_mask ? bottom_blobs[2].elempack == 1 : true;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;
    const int elempack = 4;
    const float zeros[elempack] = {0.f, 0.f, 0.f, 0.f};
    const float* zeros_ptr = zeros;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u * elempack, elempack, opt.workspace_allocator);
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int h_col = 0; h_col < outh; h_col++)
        {
            for (int w_col = 0; w_col < outw; w_col++)
            {
                int h_in = h_col * stride_h - pad_top;
                int w_in = w_col * stride_w - pad_left;
                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        float offset_h = 0.f;
                        float offset_w = 0.f;
                        float mask_ = 1.f;
                        if (offset_not_pack)
                        {
                            offset_h = offset.channel((i * kernel_w + j) * 2).row(h_col)[w_col];
                            offset_w = offset.channel((i * kernel_w + j) * 2 + 1).row(h_col)[w_col];
                        }
                        else
                        {
                            const int y_c = (i * kernel_w + j) * 2;
                            const int x_c = (i * kernel_w + j) * 2 + 1;
                            offset_h = offset.channel(y_c / offset.elempack).row(h_col)[w_col * offset.elempack + y_c % offset.elempack];
                            offset_w = offset.channel(x_c / offset.elempack).row(h_col)[w_col * offset.elempack + x_c % offset.elempack];
                        }
                        if (has_mask)
                        {
                            const Mat& mask = bottom_blobs[2];
                            if (mask_not_pack)
                            {
                                mask_ = mask.channel(i * kernel_w + j).row(h_col)[w_col];
                            }
                            else
                            {
                                const int m_c = i * kernel_w + j;
                                mask_ = mask.channel(m_c / mask.elempack).row(h_col)[w_col * mask.elempack + m_c % mask.elempack];
                            }
                        }
                        const float h_im = h_in + i * dilation_h + offset_h;
                        const float w_im = w_in + j * dilation_w + offset_w;

                        // Bilinear
                        const bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                        float w1 = 0.f;
                        float w2 = 0.f;
                        float w3 = 0.f;
                        float w4 = 0.f;
                        bool v1_cond = false;
                        bool v2_cond = false;
                        bool v3_cond = false;
                        bool v4_cond = false;
                        int v1_pos = 0;
                        int v2_pos = 0;
                        int v3_pos = 0;
                        int v4_pos = 0;
                        if (cond)
                        {
                            int h_low = floor(h_im);
                            int w_low = floor(w_im);
                            int h_high = h_low + 1;
                            int w_high = w_low + 1;

                            float lh = h_im - h_low;
                            float lw = w_im - w_low;
                            float hh = 1 - lh;
                            float hw = 1 - lw;

                            v1_cond = (h_low >= 0 && w_low >= 0);
                            v2_cond = (h_low >= 0 && w_high <= w - 1);
                            v3_cond = (h_high <= h - 1 && w_low >= 0);
                            v4_cond = (h_high <= h - 1 && w_high <= w - 1);
                            if (v1_cond)
                                v1_pos = h_low * w + w_low;
                            if (v2_cond)
                                v2_pos = h_low * w + w_high;
                            if (v3_cond)
                                v3_pos = h_high * w + w_low;
                            if (v4_cond)
                                v4_pos = h_high * w + w_high;

                            w1 = hh * hw;
                            w2 = hh * lw;
                            w3 = lh * hw;
                            w4 = lh * lw;
                        }
                        const float w1s[elempack] = {w1, w1, w1, w1};
                        const float* w1_ptr = w1s;
                        const float w2s[elempack] = {w2, w2, w2, w2};
                        const float* w2_ptr = w2s;
                        const float w3s[elempack] = {w3, w3, w3, w3};
                        const float* w3_ptr = w3s;
                        const float w4s[elempack] = {w4, w4, w4, w4};
                        const float* w4_ptr = w4s;
                        const float masks[elempack] = {mask_, mask_, mask_, mask_};
                        const float* mask_ptr = masks;

                        for (int ic = 0; ic < inch; ic++)
                        {
                            const float* data_im_ptr = bottom_blob.channel(ic);
                            __m128 _val = _mm_loadu_ps(zeros_ptr);
                            if (cond)
                            {
                                __m128 _v1 = _val;
                                __m128 _v2 = _val;
                                __m128 _v3 = _val;
                                __m128 _v4 = _val;
                                if (v1_cond)
                                    _v1 = _mm_load_ps(data_im_ptr + v1_pos * elempack);
                                if (v2_cond)
                                    _v2 = _mm_load_ps(data_im_ptr + v2_pos * elempack);
                                if (v3_cond)
                                    _v3 = _mm_load_ps(data_im_ptr + v3_pos * elempack);
                                if (v4_cond)
                                    _v4 = _mm_load_ps(data_im_ptr + v4_pos * elempack);
                                __m128 _w1 = _mm_loadu_ps(w1_ptr);
                                __m128 _w2 = _mm_loadu_ps(w2_ptr);
                                __m128 _w3 = _mm_loadu_ps(w3_ptr);
                                __m128 _w4 = _mm_loadu_ps(w4_ptr);
                                _val = _mm_comp_fmadd_ps(_v1, _w1, _val);
                                _val = _mm_comp_fmadd_ps(_v2, _w2, _val);
                                _val = _mm_comp_fmadd_ps(_v3, _w3, _val);
                                _val = _mm_comp_fmadd_ps(_v4, _w4, _val);
                            }
                            if (has_mask)
                            {
                                __m128 _mask = _mm_loadu_ps(mask_ptr);
                                _val = _mm_mul_ps(_val, _mask);
                            }
                            float* ptr = bottom_im2col.channel(ic);
                            _mm_store_ps(ptr + ((i * kernel_w + j) * size + h_col * outw + w_col) * elempack, _val);
                        }
                    }
                }
            }
        }
    }

    im2col_sgemm_pack4to16_avx512(bottom_im2col, top_blob, kernel, _bias, opt);
}
