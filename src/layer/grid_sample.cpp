// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// coord compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to coord writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "grid_sample.h"

namespace ncnn {

Grid_Sample::Grid_Sample()
{
    one_blob_only = true;
    support_inplace = false;
}

int Grid_Sample::load_param(const ParamDict& pd)
{
    resize_type = pd.get(0, 1);
    padding_mode = pd.get(1, 1);
    align_corner = pd.get(2, 0);

    if (resize_type < 1 || resize_type > 3)
    {
        NCNN_LOGE("unsupported resize type %d", resize_type);
        return -1;
    }

    if (padding_mode < 1 || padding_mode > 3)
    {
        NCNN_LOGE("unsupported padding mode %d", padding_mode);
        return -1;
    }

    return 0;
}

#if defined(__GNUC__) && defined(__powerpc__) && defined(__ALTIVEC__)
// NOTE gcc altivec optimized version produce wrong result
// so I have to disable vectorize here  --- nihui
__attribute__((optimize("no-tree-vectorize")))
#endif

// Restore normalized location to acutal image location
//   When align_corners is true:
//     Normalized location (-1, -1) points to the top-left pixel.
//     Normalized location (1, 1) points to the bottom-tight pixel.
//   When align_corners is false [default]:
//     Normalized location (-1, -1) points to the top-left pixel minus half
//     pixel coord both directions, i.e, (-0.5, -0.5) coord acutal image space.
//     Normalized location (1, 1) points to the bottom-tight pixel plus half
//     pixel coord both directions, i.e. (H - 0.5, W - 0.5) coord acutal image space.
static float
grid_sample_unormalize(int w, float coordx, int align_corner)
{
    return align_corner ? (coordx + 1) / 2.f * (w - 1) : ((coordx + 1) * w - 1) / 2.f;
}

static int border_coord(float coord, int border)
{
    return std::min(border, std::max((int)coord, 0));
}

// Reflects coordinates until they fall between low and high (inclusive).
static int reflect_coord(int coord, int low, int high)
{
    if (low == high)
    {
        return 0;
    }
    int min = low / 2;
    int span = static_cast<int>(high - low) / 2;
    coord = std::fabs(coord - min);
    // `fmod` returns same sign as `coord`, which is positive after the `fabs` above.
    int extra = std::fmod(coord, span);
    int flips = std::floor(coord / span);

    return flips % 2 ? (span - extra + min) : (extra + min);
}

// static void linear_coeffs(int w, int outw,
//                         const float *coordxs, int* xofs, float* alpha,
//                         int align_corner, int padding_mode)
// {
//     for(int dx = 0; dx < outw; dx++)
//     {
//         float fx = grid_sample_unormalize(w, coordxs[2 * dx], align_corner);

//         int sx = static_cast<int>(floor(fx));
//         fx -= sx;

//         // To Do
//         // 处理一下padding_mode的问题, 突然觉得这里处理这个不合适，
//         // 最好应该在最终的时候处理，因为最终的时候填零还是填边缘值才能确定
//         // 又想了一下，还是在这里确定吧，不然的话还是比较麻烦
//         // xs = func
//         // fx = func
//         // int padding_pivot = align_corner ? 0 : -1;
//         if(padding_mode == 1) // zeros
//         {
//             if(sx < padding_pivot || sx >= w - 1)
//             {
//                 alpha[dx * 2] = 0;
//                 alpha[dx * 2 + 1] = 0;
//             }
//             else
//             {
//                 alpha[dx * 2] = 1.f - fx;
//                 alpha[dx * 2 + 1] = fx;
//             }
//         }
//         else if(padding_mode == 2) // border
//         {
//             if (sx < 0)
//             {
//                 sx = 0;
//                 fx = 0.f;
//             }
//             if (sx >= w - 1)
//             {
//                 sx = w - 2;
//                 fx = 1.f;
//             }

//             alpha[dx * 2] = 1.f - fx;
//             alpha[dx * 2 + 1] = fx;
//         }
//         else // reflection
//         {
//             // TO DO
//             // 这个要不断反射，写个递归小代码就行应该
//         }

//         xofs[dx] = sx;
//     }
// }

static float get_coord(float x, int w, int padding_mode, int align_corner)
{
    // compute the origin coordinates and the coeffs
    float sx = grid_sample_unormalize(w, x, align_corner);

    // correct the coordinates and the coeffs according to the padding_mode
    if (padding_mode == 2) // border
    {
        sx = border_coord(sx, w - 1);
    }
    else if (padding_mode == 3) // reflection
    {
        if (align_corner)
        {
            sx = reflect_coord(sx, 0, 2 * (w - 1));
        }
        else
        {
            sx = reflect_coord(sx, -1, 2 * w - 1);
        }
    }
}

static bool in_bounds(int x, int y, int w, int h)
{
    return x >= 0 && y >= 0 && x < w && y < h;
}

static void GSample_bilinear(const Mat& src, Mat& dst, const Mat& grid,
                             int align_corner, int padding_mode)
{
    int outw = dst.w;
    int outh = dst.h;
    int w = src.w;
    int h = src.h;

    const float* srcptr = src;
    const float* dstptr = dst;

    for (int dy = 0; dy < outh; dy++)
    {
        const float* gridrowptr = grid.row(dy);
        for (int dx = 0; dx < outw; dx++)
        {
            // get the coordinate
            float x = gridrowptr[dx * 2];
            float y = gridrowptr[dx * 2 + 1];
            int x0 = get_coord(x, w, padding_mode, align_corner);
            int y0 = get_coord(y, h, padding_mode, align_corner);
            int x1 = x0 + 1;
            int y1 = y1 + 1;

            // compute the coeffs of every coord
            int a1 = x0 - static_cast<int>(std::floor(x0));
            int b1 = y0 - static_cast<int>(std::floor(y0));
            int a0 = 1 - a1;
            int b0 = 1 - b1;

            // compute the bilinear answer
            float dst0 = 0.f;

            if (in_bounds(x0, y0, w, h))
            {
                dst0 += src[y0 * w + x0] * a0 * b0;
            }
            if (in_bounds(x1, y0, w, h))
            {
                dst0 += src[y0 * w + x1] * a1 * b0;
            }
            if (in_bounds(x0, y1, w, h))
            {
                dst0 += src[y1 * w + x0] * a0 * b1;
            }
            if (in_bounds(x1, y1, w, h))
            {
                dst0 += src[y1 * w + x1] * a1 * b1;
            }

            dst[dy * outw + dx] = dst0;
        }
    }
}

int Grid_Sample::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw = grid.w;
    int outh = grid.h;

    if (dims == 3)
    {
        top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        if (resize_type == 1) // bilinear
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                GSample_bilinear(src, dst, grid, align_corner, padding_mode);
            }
        }
        else if (resize_type == 2) //nearest
        {
        }
    }

    return 0;
}

} // namespace ncnn
