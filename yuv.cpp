//
//  main.cpp
//  
//
//  Created by ji luyang on 2020/12/22.
//

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <time.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#define m64(x) (*((__m64*)x))
//  #define baseline
//  #define mmx
//  #define sse2
//  #define avx
using namespace std;

struct yuv {
    int y[1080][1920];
    int u[1080][1920];
    int v[1080][1920];
};

struct rgb {
    int r[1080][1920];
    int g[1080][1920];
    int b[1080][1920];
};
yuv yuv1, yuv2, yuv_tmp;
rgb rgb1, rgb2, rgb_tmp;

void save_yuv(yuv& yuv_file, const char* filename) {
    FILE* outfile = fopen(filename, "a+");
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j++) {
            unsigned char c = yuv_file.y[i][j];
            fwrite(&c, 1, 1, outfile);
        }
    }
    
    for (int i = 0; i < 1080; i += 2) {
        for (int j = 0; j < 1920; j += 2) {
            unsigned char c = yuv_file.u[i][j];
            fwrite(&c, 1, 1, outfile);
        }
    }
    
    for (int i = 0; i < 1080; i += 2) {
        for (int j = 0; j < 1920; j += 2) {
            unsigned char c = yuv_file.v[i][j];
            fwrite(&c, 1, 1, outfile);
        }
    }
    fclose(outfile);
}

void yuv2rgb(yuv& yuv_file, rgb& rgb_file) {
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j++) {
            rgb_file.r[i][j] = (yuv_file.y[i][j]) - 0.001 * (yuv_file.u[i][j] - 128) + 1.402 * (yuv_file.v[i][j] - 128);
            rgb_file.g[i][j] = (yuv_file.y[i][j]) - 0.344 * (yuv_file.u[i][j] - 128) - 0.714 * (yuv_file.v[i][j] - 128);
            rgb_file.b[i][j] = (yuv_file.y[i][j]) + 1.772 * (yuv_file.u[i][j] - 128) + 0.001 * (yuv_file.v[i][j] - 128);
            
            rgb_file.r[i][j] = min(255, max(0, rgb_file.r[i][j]));
            rgb_file.g[i][j] = min(255, max(0, rgb_file.g[i][j]));
            rgb_file.b[i][j] = min(255, max(0, rgb_file.b[i][j]));
            
        }
    }
}

void yuv2rgb_avx(yuv& yuv_file, rgb& rgb_file) {
    __m256 min_bound = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);
    __m256 max_bound = _mm256_set_ps(255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0);
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 8) {
            __m256i y = _mm256_loadu_si256((__m256i*)(yuv_file.y[i] + j));
            __m256i u = _mm256_loadu_si256((__m256i*)(yuv_file.u[i] + j));
            __m256i v = _mm256_loadu_si256((__m256i*)(yuv_file.v[i] + j));
            
            u = _mm256_sub_epi32(u, _mm256_set_epi32(128, 128, 128, 128, 128, 128, 128, 128));
            v = _mm256_sub_epi32(v, _mm256_set_epi32(128, 128, 128, 128, 128, 128, 128, 128));
            
            y = _mm256_cvtepi32_ps(y);
            u = _mm256_cvtepi32_ps(u);
            v = _mm256_cvtepi32_ps(v);
            
            __m256 ru = _mm256_set_ps(0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001);
            __m256 rv = _mm256_set_ps(1.402, 1.402, 1.402, 1.402, 1.402, 1.402, 1.402, 1.402);
            __m256 r2 = _mm256_mul_ps(ru, u);
            __m256 r3 = _mm256_mul_ps(rv, v);
            __m256 r = _mm256_add_ps(r3, _mm256_sub_ps(y, r2));
            
            __m256 gu = _mm256_set_ps(0.334, 0.334, 0.334, 0.334, 0.334, 0.334, 0.334, 0.334);
            __m256 gv = _mm256_set_ps(0.714, 0.714, 0.714, 0.714, 0.714, 0.714, 0.714, 0.714);
            __m256 g2 = _mm256_mul_ps(gu, u);
            __m256 g3 = _mm256_mul_ps(gv, v);
            __m256 g = _mm256_sub_ps(y, _mm256_add_ps(g2, g3));
            
            __m256 bu = _mm256_set_ps(1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.772);
            __m256 bv = _mm256_set_ps(0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001);
            __m256 b2 = _mm256_mul_ps(bu, u);
            __m256 b3 = _mm256_mul_ps(bv, v);
            __m256 b = _mm256_add_ps(y, _mm256_add_ps(b2, b3));
            
            r = _mm256_min_ps(max_bound, _mm256_max_ps(min_bound, r));
            g = _mm256_min_ps(max_bound, _mm256_max_ps(min_bound, g));
            b = _mm256_min_ps(max_bound, _mm256_max_ps(min_bound, b));
            
            __m256i r_res = _mm256_cvtps_epi32(r);
            __m256i g_res = _mm256_cvtps_epi32(g);
            __m256i b_res = _mm256_cvtps_epi32(b);
            
            _mm256_storeu_si256((__m256i*)(rgb_file.r[i] + j), r_res);
            _mm256_storeu_si256((__m256i*)(rgb_file.g[i] + j), g_res);
            _mm256_storeu_si256((__m256i*)(rgb_file.b[i] + j), b_res);
            
        }
    }
}

void yuv2rgb_sse2(yuv& yuv_file, rgb& rgb_file) {
    __m128 min_bound = _mm_set_ps(0, 0, 0, 0);
    __m128 max_bound = _mm_set_ps(255.0, 255.0, 255.0, 255.0);
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 4) {
            __m128i y = _mm_loadu_si128((__m128i*)(yuv_file.y[i] + j));
            __m128i u = _mm_loadu_si128((__m128i*)(yuv_file.u[i] + j));
            __m128i v = _mm_loadu_si128((__m128i*)(yuv_file.v[i] + j));
            
            u = _mm_sub_epi32(u, _mm_set_epi32(128, 128, 128, 128));
            v = _mm_sub_epi32(v, _mm_set_epi32(128, 128, 128, 128));
            
            y = _mm_cvtepi32_ps(y);
            u = _mm_cvtepi32_ps(u);
            v = _mm_cvtepi32_ps(v);
            
            __m128 ru = _mm_set_ps(0.001, 0.001, 0.001, 0.001);
            __m128 rv = _mm_set_ps(1.402, 1.402, 1.402, 1.402);
            __m128 r2 = _mm_mul_ps(ru, u);
            __m128 r3 = _mm_mul_ps(rv, v);
            __m128 r = _mm_add_ps(r3, _mm_sub_ps(y, r2));
            
            __m128 gu = _mm_set_ps(0.334, 0.334, 0.334, 0.334);
            __m128 gv = _mm_set_ps(0.714, 0.714, 0.714, 0.714);
            __m128 g2 = _mm_mul_ps(gu, u);
            __m128 g3 = _mm_mul_ps(gv, v);
            __m128 g = _mm_sub_ps(y, _mm_add_ps(g2, g3));
            
            __m128 bu = _mm_set_ps(1.772, 1.772, 1.772, 1.772);
            __m128 bv = _mm_set_ps(0.001, 0.001, 0.001, 0.001);
            __m128 b2 = _mm_mul_ps(bu, u);
            __m128 b3 = _mm_mul_ps(bv, v);
            __m128 b = _mm_add_ps(y, _mm_add_ps(b2, b3));
            
            r = _mm_min_ps(max_bound, _mm_max_ps(min_bound, r));
            g = _mm_min_ps(max_bound, _mm_max_ps(min_bound, g));
            b = _mm_min_ps(max_bound, _mm_max_ps(min_bound, b));
            
            __m128i r_res = _mm_cvtps_epi32(r);
            __m128i g_res = _mm_cvtps_epi32(g);
            __m128i b_res = _mm_cvtps_epi32(b);
            
            _mm_storeu_si128((__m128i*)(rgb_file.r[i] + j), r_res);
            _mm_storeu_si128((__m128i*)(rgb_file.g[i] + j), g_res);
            _mm_storeu_si128((__m128i*)(rgb_file.b[i] + j), b_res);
            
        }
    }

}

void yuv2rgb_mmx(yuv& yuv_file, rgb& rgb_file) {
    __m64 v_of_r = _mm_set1_pi32(45);  //5
    __m64 const_of_r = _mm_set1_pi32(179);
    __m64 u_of_g = _mm_set1_pi32(11);   //5
    __m64 v_of_g = _mm_set1_pi32(3); //2
    __m64 const_of_g = _mm_set1_pi32(135);
    __m64 u_of_b = _mm_set1_pi32(7);    //2
    __m64 const_of_b = _mm_set1_pi32(227);
    
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 2) {
            //if (!i && j==2) cout<<yuv_file.y[i][j] << " " << yuv_file.y[i][j+1]<<endl;
            __m64 r1 = m64((yuv_file.y[i] + j));
            __m64 r3 = _mm_mullo_pi16(m64((yuv_file.v[i] + j)), v_of_r);
            r3 = _mm_srli_pi32(r3, 5);

            __m64 g1 = m64((yuv_file.y[i] + j));
            __m64 g2 = _mm_mullo_pi16(m64((yuv_file.u[i] + j)), u_of_g);
            g2 = _mm_srli_pi32(g2, 5);
            __m64 g3 = _mm_mullo_pi16(m64((yuv_file.v[i] + j)), v_of_g);
            g3 = _mm_srli_pi32(g3, 2);
            
            __m64 b1 = m64((yuv_file.y[i] + j));
            __m64 b2 = _mm_mullo_pi16(m64((yuv_file.u[i] + j)), u_of_b);
            b2 = _mm_srli_pi32(b2, 2);
            
            m64((rgb_file.r[i] + j)) = _mm_sub_pi32(_mm_add_pi32(r1, r3), const_of_r);
            m64((rgb_file.g[i] + j)) = _mm_add_pi32(_mm_sub_pi32(_mm_sub_pi32(g1, g2), g3), const_of_g);
            m64((rgb_file.b[i] + j)) = _mm_sub_pi32(_mm_add_pi32(b1, b2), const_of_b);
            
        }
    }
    
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j++) {
            rgb_file.r[i][j] = min(255, max(0, rgb_file.r[i][j]));
            rgb_file.g[i][j] = min(255, max(0, rgb_file.g[i][j]));
            rgb_file.b[i][j] = min(255, max(0, rgb_file.b[i][j]));
            //cout<<i<<" "<<j<<endl;
            //cout<<rgb_file.r[i][j]<<" "<<rgb_file.g[i][j]<<" "<<rgb_file.b[i][j]<<endl;
        }
    }
}

void rgb2yuv(rgb& rgb_file, yuv& yuv_file, int a = 256) {
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j++) {
            int r = rgb_file.r[i][j] * a / 256;
            int g = rgb_file.g[i][j] * a / 256;
            int b = rgb_file.b[i][j] * a / 256;
            yuv_file.y[i][j] = 0.299 * r + 0.587 * g + 0.114 * b;
            yuv_file.u[i][j] = -0.169 * r - 0.331 * g + 0.5 * b + 128;
            yuv_file.v[i][j] = 0.5 * r - 0.419 * g - 0.081 * b + 128;
            
            yuv_file.y[i][j] = min(255, max(0, yuv_file.y[i][j]));
            yuv_file.u[i][j] = min(255, max(0, yuv_file.u[i][j]));
            yuv_file.v[i][j] = min(255, max(0, yuv_file.v[i][j]));
        }
    }
}

void rgb2yuv_avx(rgb& rgb_file, yuv& yuv_file, int a = 256) {
    float fa = (float)a / 256;
    __m256 ka = _mm256_set_ps(fa, fa, fa, fa, fa, fa, fa, fa);
    __m256 bias = _mm256_set_ps(128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0);
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 8) {
            __m256 r = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file.r[i] + j)));
            __m256 g = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file.g[i] + j)));
            __m256 b = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file.b[i] + j)));
            
            r = _mm256_mul_ps(ka, r);
            g = _mm256_mul_ps(ka, g);
            b = _mm256_mul_ps(ka, b);
            
            __m256 yr = _mm256_mul_ps(_mm256_set_ps(0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299), r);
            __m256 yg = _mm256_mul_ps(_mm256_set_ps(0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587, 0.587), g);
            __m256 yb = _mm256_mul_ps(_mm256_set_ps(0.114, 0.114, 0.114, 0.114, 0.114, 0.114, 0.114, 0.114), b);
            __m256 ur = _mm256_mul_ps(_mm256_set_ps(0.169, 0.169, 0.169, 0.169, 0.169, 0.169, 0.169, 0.169), r);
            __m256 ug = _mm256_mul_ps(_mm256_set_ps(0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331, 0.331), g);
            __m256 ub = _mm256_mul_ps(_mm256_set_ps(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), b);
            __m256 vr = _mm256_mul_ps(_mm256_set_ps(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), r);
            __m256 vg = _mm256_mul_ps(_mm256_set_ps(0.419, 0.419, 0.419, 0.419, 0.419, 0.419, 0.419, 0.419), g);
            __m256 vb = _mm256_mul_ps(_mm256_set_ps(0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081), b);
            
            __m256 y = _mm256_add_ps(yr, _mm256_add_ps(yg, yb));
            __m256 u = _mm256_add_ps(ub, _mm256_sub_ps(bias, _mm256_add_ps(ur, ug)));
            __m256 v = _mm256_add_ps(vr, _mm256_sub_ps(bias, _mm256_add_ps(vg, vb)));
            
            __m256i y_res = _mm256_cvtps_epi32(y);
            __m256i u_res = _mm256_cvtps_epi32(u);
            __m256i v_res = _mm256_cvtps_epi32(v);
            
            _mm256_storeu_si256((__m256i*)(yuv_file.y[i] + j), y_res);
            _mm256_storeu_si256((__m256i*)(yuv_file.u[i] + j), u_res);
            _mm256_storeu_si256((__m256i*)(yuv_file.v[i] + j), v_res);
            
        }
    }
    
}

void rgb2yuv_sse2(rgb& rgb_file, yuv& yuv_file, int a = 256) {
    float fa = (float)a / 256;
    __m128 ka = _mm_set_ps(fa, fa, fa, fa);
    __m128 bias = _mm_set_ps(128.0, 128.0, 128.0, 128.0);
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 4) {
            __m128 r = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file.r[i] + j)));
            __m128 g = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file.g[i] + j)));
            __m128 b = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file.b[i] + j)));
            
            r = _mm_mul_ps(ka, r);
            g = _mm_mul_ps(ka, g);
            b = _mm_mul_ps(ka, b);
            
            __m128 yr = _mm_mul_ps(_mm_set_ps(0.299, 0.299, 0.299, 0.299), r);
            __m128 yg = _mm_mul_ps(_mm_set_ps(0.587, 0.587, 0.587, 0.587), g);
            __m128 yb = _mm_mul_ps(_mm_set_ps(0.114, 0.114, 0.114, 0.114), b);
            __m128 ur = _mm_mul_ps(_mm_set_ps(0.169, 0.169, 0.169, 0.169), r);
            __m128 ug = _mm_mul_ps(_mm_set_ps(0.331, 0.331, 0.331, 0.331), g);
            __m128 ub = _mm_mul_ps(_mm_set_ps(0.5, 0.5, 0.5, 0.5), b);
            __m128 vr = _mm_mul_ps(_mm_set_ps(0.5, 0.5, 0.5, 0.5), r);
            __m128 vg = _mm_mul_ps(_mm_set_ps(0.419, 0.419, 0.419, 0.419), g);
            __m128 vb = _mm_mul_ps(_mm_set_ps(0.081, 0.081, 0.081, 0.081), b);
            
            __m128 y = _mm_add_ps(yr, _mm_add_ps(yg, yb));
            __m128 u = _mm_add_ps(ub, _mm_sub_ps(bias, _mm_add_ps(ur, ug)));
            __m128 v = _mm_add_ps(vr, _mm_sub_ps(bias, _mm_add_ps(vg, vb)));
            
            __m128i y_res = _mm_cvtps_epi32(y);
            __m128i u_res = _mm_cvtps_epi32(u);
            __m128i v_res = _mm_cvtps_epi32(v);
            
            _mm_storeu_si128((__m128i*)(yuv_file.y[i] + j), y_res);
            _mm_storeu_si128((__m128i*)(yuv_file.u[i] + j), u_res);
            _mm_storeu_si128((__m128i*)(yuv_file.v[i] + j), v_res);
            
        }
    }
}

void rgb2yuv_mmx(rgb& rgb_file, yuv& yuv_file, int a = 256) {
    __m64 bias = _mm_set1_pi32(128);
    __m64 ka = _mm_set1_pi32(a);
    __m64 r1 = _mm_set1_pi32(153);  //9
    __m64 g1 = _mm_set1_pi32(75);  //7

    __m64 r2 = _mm_set1_pi32(173);  //10
    __m64 g2 = _mm_set1_pi32(21);  //6
    __m64 g3 = _mm_set1_pi32(27);  //6
    __m64 b3 = _mm_set1_pi32(83);   //10
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 2) {
            __m64 r = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file.r[i] + j)), ka), 8);
            __m64 g = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file.g[i] + j)), ka), 8);
            __m64 b = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file.b[i] + j)), ka), 8);
            
            __m64 y1 = _mm_srli_pi32(_mm_mullo_pi16(r1, r), 9);
            __m64 y2 = _mm_srli_pi32(_mm_mullo_pi16(g1, g), 7);
            __m64 y3 = _mm_srli_pi32(b, 8);
            __m64 u1 = _mm_srli_pi32(_mm_mullo_pi16(r2, r), 10);
            __m64 u2 = _mm_srli_pi32(_mm_mullo_pi16(g2, g), 6);
            __m64 u3 = _mm_srli_pi32(b, 1);
            __m64 v1 = _mm_srli_pi32(r, 1);
            __m64 v2 = _mm_srli_pi32(_mm_mullo_pi16(g3, g), 6);
            __m64 v3 = _mm_srli_pi32(_mm_mullo_pi16(b3, b), 10);
            
            m64((yuv_file.y[i] + j)) = _mm_add_pi32(_mm_add_pi32(y1, y2), y3);
            m64((yuv_file.u[i] + j)) = _mm_add_pi32(_mm_sub_pi32(_mm_sub_pi32(u3, u2), u1), bias);
            m64((yuv_file.v[i] + j)) = _mm_add_pi32(_mm_sub_pi32(_mm_sub_pi32(v1, v2), v3), bias);
            
        }
    }
    
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j++) {
            yuv_file.y[i][j] = min(255, max(0, yuv_file.y[i][j]));
            yuv_file.u[i][j] = min(255, max(0, yuv_file.u[i][j]));
            yuv_file.v[i][j] = min(255, max(0, yuv_file.v[i][j]));
        }
    }
}

void merge_rgb(rgb& rgb_file1, rgb& rgb_file2, rgb& rgb_tmp, int a) {
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j++) {
            rgb_tmp.r[i][j] = (rgb_file1.r[i][j] * a + rgb_file2.r[i][j] * (256 - a)) / 256;
            rgb_tmp.g[i][j] = (rgb_file1.g[i][j] * a + rgb_file2.g[i][j] * (256 - a)) / 256;
            rgb_tmp.b[i][j] = (rgb_file1.b[i][j] * a + rgb_file2.b[i][j] * (256 - a)) / 256;
        }
    }
}

void merge_rgb_avx(rgb& rgb_file1, rgb& rgb_file2, rgb& rgb_tmp, int a) {
    float fa = (float)a / 256.0;
    float fb = 1 - fa;
    __m256 k1 = _mm256_set_ps(fa, fa, fa, fa, fa, fa, fa, fa);
    __m256 k2 = _mm256_set_ps(fb, fb, fb, fb, fb, fb, fb, fb);
    
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 8) {
            __m256 r1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file1.r[i] + j)));
            __m256 g1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file1.g[i] + j)));
            __m256 b1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file1.b[i] + j)));
            __m256 r2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file2.r[i] + j)));
            __m256 g2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file2.g[i] + j)));
            __m256 b2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(rgb_file2.b[i] + j)));
            
            __m256 r = _mm256_add_ps(_mm256_mul_ps(k1, r1), _mm256_mul_ps(k2, r2));
            __m256 g = _mm256_add_ps(_mm256_mul_ps(k1, g1), _mm256_mul_ps(k2, g2));
            __m256 b = _mm256_add_ps(_mm256_mul_ps(k1, b1), _mm256_mul_ps(k2, b2));
            
            __m256i r_res = _mm256_cvtps_epi32(r);
            __m256i g_res = _mm256_cvtps_epi32(g);
            __m256i b_res = _mm256_cvtps_epi32(b);
            
            _mm256_storeu_si256((__m256i*)(rgb_tmp.r[i] + j), r_res);
            _mm256_storeu_si256((__m256i*)(rgb_tmp.g[i] + j), g_res);
            _mm256_storeu_si256((__m256i*)(rgb_tmp.b[i] + j), b_res);
            
        }
    }
    
}

void merge_rgb_sse2(rgb& rgb_file1, rgb& rgb_file2, rgb& rgb_tmp, int a) {
    float fa = (float)a / 256.0;
    float fb = 1 - fa;
    __m128 k1 = _mm_set_ps(fa, fa, fa, fa);
    __m128 k2 = _mm_set_ps(fb, fb, fb, fb);

    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 4) {
            __m128 r1 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file1.r[i] + j)));
            __m128 g1 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file1.g[i] + j)));
            __m128 b1 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file1.b[i] + j)));
            __m128 r2 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file2.r[i] + j)));
            __m128 g2 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file2.g[i] + j)));
            __m128 b2 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)(rgb_file2.b[i] + j)));
            
            __m128 r = _mm_add_ps(_mm_mul_ps(k1, r1), _mm_mul_ps(k2, r2));
            __m128 g = _mm_add_ps(_mm_mul_ps(k1, g1), _mm_mul_ps(k2, g2));
            __m128 b = _mm_add_ps(_mm_mul_ps(k1, b1), _mm_mul_ps(k2, b2));
            
            __m128i r_res = _mm_cvtps_epi32(r);
            __m128i g_res = _mm_cvtps_epi32(g);
            __m128i b_res = _mm_cvtps_epi32(b);
            
            _mm_storeu_si128((__m128i*)(rgb_tmp.r[i] + j), r_res);
            _mm_storeu_si128((__m128i*)(rgb_tmp.g[i] + j), g_res);
            _mm_storeu_si128((__m128i*)(rgb_tmp.b[i] + j), b_res);
            
        }
    }
}

void merge_rgb_mmx(rgb& rgb_file1, rgb& rgb_file2, rgb& rgb_tmp, int a) {
    __m64 k1 = _mm_set1_pi32(a);
    __m64 k2 = _mm_set1_pi32(256 - a);
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j += 2) {
            __m64 r1 = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file1.r[i] + j)), k1), 8);
            __m64 g1 = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file1.g[i] + j)), k1), 8);
            __m64 b1 = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file1.b[i] + j)), k1), 8);
            __m64 r2 = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file2.r[i] + j)), k2), 8);
            __m64 g2 = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file2.g[i] + j)), k2), 8);
            __m64 b2 = _mm_srli_pi32(_mm_mullo_pi16(m64((rgb_file2.b[i] + j)), k2), 8);
            __m64 r = _mm_add_pi32(r1, r2);
            __m64 g = _mm_add_pi32(g1, g2);
            __m64 b = _mm_add_pi32(b1, b2);
            
            m64((rgb_tmp.r[i] + j)) = r;
            m64((rgb_tmp.g[i] + j)) = g;
            m64((rgb_tmp.b[i] + j)) = b;
            
        }
    }
}

void run1_baseline() {
    yuv2rgb(yuv1, rgb1);
    const char* filename = "fade.yuv";
    for (int a = 1; a <= 255; a += 3) {
        rgb2yuv(rgb1, yuv_tmp, a);
#ifdef baseline
        save_yuv(yuv_tmp, filename);
#endif
    }
}

void run1_mmx() {
    yuv2rgb_mmx(yuv1, rgb1);
    const char* filename = "fade_mmx.yuv";
    for (int a = 1; a <= 255; a += 3) {
        rgb2yuv_mmx(rgb1, yuv_tmp, a);
#ifdef mmx
        save_yuv(yuv_tmp, filename);
#endif
    }
}

void run1_sse2() {
    yuv2rgb_sse2(yuv1, rgb1);
    const char* filename = "fade_sse2.yuv";
    for (int a = 1; a <= 255; a += 3) {
        rgb2yuv_sse2(rgb1, yuv_tmp, a);
#ifdef sse2
        save_yuv(yuv_tmp, filename);
#endif
    }
}

void run1_avx() {
    yuv2rgb_avx(yuv1, rgb1);
    const char* filename = "fade_avx.yuv";
    for (int a = 1; a <= 255; a += 3) {
        rgb2yuv_avx(rgb1, yuv_tmp, a);
#ifdef avx
        save_yuv(yuv_tmp, filename);
#endif
    }
}

void run2_baseline() {
    yuv2rgb(yuv1, rgb1);
    yuv2rgb(yuv2, rgb2);
    const char* filename = "merge.yuv";
    for (int a = 1; a <= 255; a += 3) {
        merge_rgb(rgb1, rgb2, rgb_tmp, a);
        rgb2yuv(rgb_tmp, yuv_tmp);
#ifdef baseline
        save_yuv(yuv_tmp, filename);
#endif
    }
}

void run2_mmx() {
    yuv2rgb_mmx(yuv1, rgb1);
    yuv2rgb_mmx(yuv2, rgb2);
    const char* filename = "merge_mmx.yuv";
    for (int a = 1; a <= 255; a += 3) {
        merge_rgb_mmx(rgb1, rgb2, rgb_tmp, a);
        rgb2yuv_mmx(rgb_tmp, yuv_tmp);
#ifdef mmx
        save_yuv(yuv_tmp, filename);
#endif
    }
}

void run2_sse2() {
    yuv2rgb_sse2(yuv1, rgb1);
    yuv2rgb_sse2(yuv2, rgb2);
    const char* filename = "merge_sse2.yuv";
    for (int a = 1; a <= 255; a += 3) {
        merge_rgb_sse2(rgb1, rgb2, rgb_tmp, a);
        rgb2yuv_sse2(rgb_tmp, yuv_tmp);
#ifdef sse2
        save_yuv(yuv_tmp, filename);
#endif
    }
}

void run2_avx() {
    yuv2rgb_avx(yuv1, rgb1);
    yuv2rgb_avx(yuv2, rgb2);
    const char* filename = "merge_avx.yuv";
    for (int a = 1; a <= 255; a += 3) {
        merge_rgb_avx(rgb1, rgb2, rgb_tmp, a);
        rgb2yuv_avx(rgb_tmp, yuv_tmp);
#ifdef avx
        save_yuv(yuv_tmp, filename);
#endif
    }
}

void print_time(int t1, int t2, int t3, int t4, int t5) {
    cout << "Process Baseline: \t" << (t2 - t1) / 1000000.0 << "s" << endl;
    cout << "Process with mmx: \t" << (t3 - t2) / 1000000.0 << "s" << endl;
    cout << "Process with sse2: \t" << (t4 - t3) / 1000000.0 << "s" << endl;
    cout << "Process with avx: \t" << (t5 - t4) / 1000000.0 << "s" << endl;
}
int main(int argc, char* argv[]) {
    if (argc != 2 || strcmp(argv[0], "./yuv") || strcmp(argv[1], "1") && strcmp(argv[1], "2")) {
        printf("Invalid input. Please input commands like ./yuv 1 or 2.\n");
        exit(0);
    }
    FILE* f1 = fopen("demo/dem1.yuv", "rb");
    FILE* f2 = fopen("demo/dem2.yuv", "rb");
    
    for (int i = 0; i < 1080; i++) {
        for (int j = 0; j < 1920; j++) {
            unsigned char c;
            fread(&c, 1, 1, f1);
            yuv1.y[i][j] = (int)c;
            fread(&c, 1, 1, f2);
            yuv2.y[i][j] = (int)c;
        }
    }
    for (int i = 0; i < 1080; i += 2) {
        for (int j = 0; j < 1920; j += 2) {
            unsigned char c;
            fread(&c, 1, 1, f1);
            
            yuv1.u[i][j] = (int)c;
            yuv1.u[i][j + 1] = (int)c;
            yuv1.u[i + 1][j] = (int)c;
            yuv1.u[i + 1][j + 1] = (int)c;
            
            fread(&c, 1, 1, f2);
            
            yuv2.u[i][j] = (int)c;
            yuv2.u[i][j + 1] = (int)c;
            yuv2.u[i + 1][j] = (int)c;
            yuv2.u[i + 1][j + 1] = (int)c;
        }
    }
    
    for (int i = 0; i < 1080; i += 2) {
        for (int j = 0; j < 1920; j += 2) {
            unsigned char c;
            fread(&c, 1, 1, f1);
            
            yuv1.v[i][j] = (int)c;
            yuv1.v[i][j + 1] = (int)c;
            yuv1.v[i + 1][j] = (int)c;
            yuv1.v[i + 1][j + 1] = (int)c;
            
            fread(&c, 1, 1, f2);
            
            yuv2.v[i][j] = (int)c;
            yuv2.v[i][j + 1] = (int)c;
            yuv2.v[i + 1][j] = (int)c;
            yuv2.v[i + 1][j + 1] = (int)c;
        }
    }
    fclose(f1);
    fclose(f2);
    int t1, t2, t3, t4, t5;
    if (!strcmp(argv[1], "1")) {
        t1 = clock();
        run1_baseline();
        t2 = clock();
        run1_mmx();
        t3 = clock();
        run1_sse2();
        t4 = clock();
        run1_avx();
        t5 = clock();
        print_time(t1, t2, t3, t4, t5);
    } else if (!strcmp(argv[1], "2")) {
        t1 = clock();
        run2_baseline();
        t2 = clock();
        run2_mmx();
        t3 = clock();
        run2_sse2();
        t4 = clock();
        run2_avx();
        t5 = clock();
        print_time(t1, t2, t3, t4, t5);
    }
    return 0;
}
