/**
 * @file cosine_neon.c
 * @brief High-performance ARM NEON-optimized Cosine Similarity for float16 embeddings.
 * Optimized for 384-dimensional dense vectors (e.g., MiniLM-L6-v2) on ARM v8-A/v8.2-A+ Cortex processors.
 * Target execution time: <0.3ms on Cortex-A55 cores.
 */

#include <stdio.h>
#include <math.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

/**
 * @brief Calculates the cosine similarity between two 384-dimensional float16 vectors using NEON intrinsics.
 * 
 * @param a Pointer to the first float16 array (384 elements, 16-byte aligned).
 * @param b Pointer to the second float16 array (384 elements, 16-byte aligned).
 * @return float Cosine similarity score in range [-1.0, 1.0].
 */
float cosine_similarity_neon_f16(const __fp16 *a, const __fp16 *b) {
    // 384 dimensions / 8 elements per NEON float16x8 register = 48 iterations.
    const int dims = 384;
    const int loop_count = dims / 8;
    
    // Accumulator registers for dot product, magnitude A, and magnitude B
    float16x8_t dot_accum = vdupq_n_f16(0.0f16);
    float16x8_t mag_a_accum = vdupq_n_f16(0.0f16);
    float16x8_t mag_b_accum = vdupq_n_f16(0.0f16);
    
    for (int i = 0; i < loop_count; i++) {
        // Load 8 float16 values from each array
        float16x8_t va = vld1q_f16(&a[i * 8]);
        float16x8_t vb = vld1q_f16(&b[i * 8]);
        
        // Fused Multiply-Accumulate / Multiply-Add operations:
        // dot_accum += va * vb
        dot_accum = vfmaq_f16(dot_accum, va, vb);
        
        // mag_a_accum += va * va
        mag_a_accum = vfmaq_f16(mag_a_accum, va, va);
        
        // mag_b_accum += vb * vb
        mag_b_accum = vfmaq_f16(mag_b_accum, vb, vb);
    }
    
    // Pairwise reduction of float16x8 vectors down to a single float32 accumulator
    // (A55/A76 are highly optimized for widening and horizontal addition)
    float dot_sum = (float)vaddvq_f16(dot_accum);
    float mag_a_sum = (float)vaddvq_f16(mag_a_accum);
    float mag_b_sum = (float)vaddvq_f16(mag_b_accum);
    
    if (mag_a_sum <= 0.0f || mag_b_sum <= 0.0f) {
        return 0.0f;
    }
    
    return dot_sum / (sqrtf(mag_a_sum) * sqrtf(mag_b_sum));
}

#else

// Portable scalar fallback for compilation/testing on x86_64, non-NEON or legacy systems.
float cosine_similarity_neon_f16(const float *a, const float *b) {
    const int dims = 384;
    double dot_sum = 0.0;
    double mag_a_sum = 0.0;
    double mag_b_sum = 0.0;
    
    for (int i = 0; i < dims; i++) {
        float val_a = a[i];
        float val_b = b[i];
        dot_sum += (double)val_a * val_b;
        mag_a_sum += (double)val_a * val_a;
        mag_b_sum += (double)val_b * val_b;
    }
    
    if (mag_a_sum <= 0.0 || mag_b_sum <= 0.0) {
        return 0.0f;
    }
    
    return (float)(dot_sum / (sqrt(mag_a_sum) * sqrt(mag_b_sum)));
}

#endif

// Exposed float32 wrapper for easier JNI mapping if float16 arguments are mapped from JNI arrays
float cosine_similarity_neon_f32(const float *a, const float *b, int dims) {
#if defined(__ARM_NEON)
    // 128-bit vector registers hold 4 float32 values
    int loop_count = dims / 4;
    float32x4_t dot_accum = vdupq_n_f32(0.0f);
    float32x4_t mag_a_accum = vdupq_n_f32(0.0f);
    float32x4_t mag_b_accum = vdupq_n_f32(0.0f);
    
    for (int i = 0; i < loop_count; i++) {
        float32x4_t va = vld1q_f32(&a[i * 4]);
        float32x4_t vb = vld1q_f32(&b[i * 4]);
        
        dot_accum = vmlaq_f32(dot_accum, va, vb);
        mag_a_accum = vmlaq_f32(mag_a_accum, va, va);
        mag_b_accum = vmlaq_f32(mag_b_accum, vb, vb);
    }
    
    float dot_sum = vaddvq_f32(dot_accum);
    float mag_a_sum = vaddvq_f32(mag_a_accum);
    float mag_b_sum = vaddvq_f32(mag_b_accum);
    
    // Handle remainders if dimensions are not a multiple of 4
    for (int i = loop_count * 4; i < dims; i++) {
        dot_sum += a[i] * b[i];
        mag_a_sum += a[i] * a[i];
        mag_b_sum += b[i] * b[i];
    }
#else
    float dot_sum = 0.0f;
    float mag_a_sum = 0.0f;
    float mag_b_sum = 0.0f;
    for (int i = 0; i < dims; i++) {
        dot_sum += a[i] * b[i];
        mag_a_sum += a[i] * a[i];
        mag_b_sum += b[i] * b[i];
    }
#endif

    if (mag_a_sum <= 0.0f || mag_b_sum <= 0.0f) {
        return 0.0f;
    }
    return dot_sum / (sqrtf(mag_a_sum) * sqrtf(mag_b_sum));
}
