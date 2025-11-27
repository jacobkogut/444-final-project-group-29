/*
 * vq_pipeline.c
 *
 * Created on: Nov 26, 2025
 * Author: williamzimmerman
 */

#include "vq_pipeline.h"
#include "encoder.h"
#include "decoder.h"
#include "encoder_data.h"
#include "decoder_data.h"
#include "codebook.h"
#include <float.h>
#include <math.h>
#include "arm_math.h" // <<< MANDATORY FOR SIMD FUNCTIONS

#define MU 255.0f
#define LN_1_MU 5.54517744f // ln(1 + 255)
#define INV_MU  0.00392156f // 1 / 255

// AI Handles
static ai_handle encoder_handle = AI_HANDLE_NULL;
static ai_handle decoder_handle = AI_HANDLE_NULL;

// === FIXED: Consolidated Global Buffers (Only one definition block) ===
// Encoder Out / VQ In: 32 channels * 64 time steps = 2048 floats
static AI_ALIGNED(4) ai_float latent_buffer[2048];
// VQ Out / Decoder In
static AI_ALIGNED(4) ai_float quantized_buffer[2048];
// Temp buffer for Mu-Law conversion/storage
static AI_ALIGNED(4) ai_float mulaw_buffer[512];

// Activation RAM (Workspace for the models)
static AI_ALIGNED(4) ai_u8 activations_enc[AI_ENCODER_DATA_ACTIVATIONS_SIZE];
static AI_ALIGNED(4) ai_u8 activations_dec[AI_DECODER_DATA_ACTIVATIONS_SIZE];

static float codebook_norms_sq[512];


void VQ_Init(void) {
    ai_error err;

    // Create Encoder
    const ai_handle acts_enc[] = { activations_enc };
    err = ai_encoder_create_and_init(&encoder_handle, acts_enc, NULL);
    if (err.type != AI_ERROR_NONE) {
        while(1); // Trap error
    }

    // Create Decoder
    const ai_handle acts_dec[] = { activations_dec };
    err = ai_decoder_create_and_init(&decoder_handle, acts_dec, NULL);
    if (err.type != AI_ERROR_NONE) {
        while(1); // Trap error
    }

    for(int c = 0; c < 512; c++) {
            const float* pVecB = &CODEBOOK[c * 32];

            // arm_power_f32 computes the sum of squares: ||B||^2 = sum(B_i^2)
            arm_power_f32(
                pVecB,     // Input vector B
                32,        // Block size (32 elements)
                &codebook_norms_sq[c] // Output: ||B||^2
            );
        }
}

// =====================================================================
//                 SIMD VQ Search and Lookup Helpers
// =====================================================================

static void VQ_Find_Indices_Optimized(float* latent_in, uint16_t* indices_out) {
    // 64 is the number of time steps (vectors) to process
    for (int t = 0; t < 64; t++) {
        int best_idx = 0;
        float min_dist_sq = FLT_MAX; // Working with squared distance

        // 1. Calculate ||A||^2 (Norm of Latent Vector) using SIMD
        float latent_norm_sq = 0.0f;
        const float* pVecA = &latent_in[t*32];

        // arm_power_f32 computes ||A||^2 = sum(A_i^2)
        arm_power_f32(pVecA, 32, &latent_norm_sq);

        // 2. Search using the Dot Product Identity
        for (int c = 0; c < 512; c++) {
            float dot_product = 0.0f;
            const float* pVecB = &CODEBOOK[c*32];

            // 🌟 STEP 2A: Calculate A * B using SIMD 🌟
            // arm_dot_prod_f32 is heavily optimized for the M4 DSP
            arm_dot_prod_f32(
                pVecA,          // Input Vector A
                pVecB,          // Codebook Vector B
                32,             // Block size
                &dot_product    // Output: A • B
            );

            // 3. STEP 2B: Calculate the final squared distance
            // ||A - B||^2 = ||A||^2 + ||B||^2 - 2(A • B)
            float current_dist_sq = latent_norm_sq
                                  + codebook_norms_sq[c]
                                  - 2.0f * dot_product;

            // Note: Since all terms are positive, we don't need to take the sqrt.
            if (current_dist_sq < min_dist_sq) {
                min_dist_sq = current_dist_sq;
                best_idx = c;
            }
        }
        indices_out[t] = (uint16_t)best_idx;
    }
}


// Helper 2: SIMD-Optimized Index to Vector Lookup (Used in Rx_Process)
static void VQ_Indices_To_Vectors(uint16_t* indices_in, float* quantized_out) {
    for (int t = 0; t < 64; t++) {
        int idx = indices_in[t];
        if(idx >= 512) idx = 0;

        const float* pCodeVec = &CODEBOOK[idx * 32];
        float* pDestVec = &quantized_out[t * 32];

        // ✅ SIMD CALL for fast copy
        arm_copy_f32(pCodeVec, pDestVec, 32);
    }
}


// =====================================================================
//               Mu-Law Functions (Corrected and Consolidated)
// =====================================================================

void MuLaw_Encode_Buffer(float* input, float* output, int length) {
    for (int i = 0; i < length; i++) {
        float x = input[i];
        if (x > 1.0f) x = 1.0f;
        if (x < -1.0f) x = -1.0f;

        float sign = (x < 0.0f) ? -1.0f : 1.0f;
        float abs_x = (x < 0.0f) ? -x : x;

        output[i] = sign * (logf(1.0f + MU * abs_x) / LN_1_MU);
    }
}

void MuLaw_Decode_Buffer(float* input, float* output, int length) {
    for (int i = 0; i < length; i++) {
        float y = input[i];

        float sign = (y < 0.0f) ? -1.0f : 1.0f;
        float abs_y = (y < 0.0f) ? -y : y;

        float numerator = powf(1.0f + MU, abs_y) - 1.0f;

        output[i] = sign * INV_MU * numerator;
    }
}


// =====================================================================
//                   MAIN TRANSMITTER/RECEIVER LOGIC
// =====================================================================

void Tx_Process(float *input_audio, uint16_t *output_indices) {
    // 1. Pre-processing: Raw -> MuLaw
    MuLaw_Encode_Buffer(input_audio, mulaw_buffer, 512);

    // 2. Encoder: MuLaw -> Latent
    ai_buffer ai_in[AI_ENCODER_IN_NUM];
    ai_buffer ai_out[AI_ENCODER_OUT_NUM];

    ai_in[0] = *ai_encoder_inputs_get(encoder_handle, NULL);
    ai_in[0].data = AI_HANDLE_PTR(mulaw_buffer);

    ai_out[0] = *ai_encoder_outputs_get(encoder_handle, NULL);
    ai_out[0].data = AI_HANDLE_PTR(latent_buffer);

    ai_encoder_run(encoder_handle, ai_in, ai_out);

    // 3. VQ Search: Latent -> Indices (SIMD)
    VQ_Find_Indices_Optimized(latent_buffer, output_indices);
}


void Rx_Process(uint16_t *input_indices, float *output_audio) {
    // 1. Lookup: Indices -> Quantized Vectors (SIMD)
    VQ_Indices_To_Vectors(input_indices, quantized_buffer);

    // 2. Decoder: Quantized -> MuLaw Audio
    ai_buffer dec_in[AI_DECODER_IN_NUM];
    ai_buffer dec_out[AI_DECODER_OUT_NUM];

    dec_in[0] = *ai_decoder_inputs_get(decoder_handle, NULL);
    dec_in[0].data = AI_HANDLE_PTR(quantized_buffer);

    dec_out[0] = *ai_decoder_outputs_get(decoder_handle, NULL);
    dec_out[0].data = AI_HANDLE_PTR(mulaw_buffer);

    ai_decoder_run(decoder_handle, dec_in, dec_out);

    // 3. Post-processing: MuLaw -> Linear Audio
    MuLaw_Decode_Buffer(mulaw_buffer, output_audio, 512);
}
