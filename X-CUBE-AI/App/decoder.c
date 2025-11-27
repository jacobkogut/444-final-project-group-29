/**
  ******************************************************************************
  * @file    decoder.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-11-27T01:23:50-0500
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "decoder.h"
#include "decoder_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_decoder
 
#undef AI_DECODER_MODEL_SIGNATURE
#define AI_DECODER_MODEL_SIGNATURE     "0x5b4a5abd892a0876b2ceb826b015a40d"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-11-27T01:23:50-0500"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_DECODER_N_BATCHES
#define AI_DECODER_N_BATCHES         (1)

static ai_ptr g_decoder_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_decoder_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  latent_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2048, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  latent_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  convolution_upsample_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4064, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  convolution_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  convolution_1_upsample_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16320, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  convolution_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  add_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  relu_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  convolution_2_upsample_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16352, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  convolution_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  audio_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 512, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  convolution_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  convolution_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  convolution_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  convolution_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  convolution_2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  convolution_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  convolution_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  convolution_1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  convolution_2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  add_1_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 256, 1), AI_STRIDE_INIT(4, 4, 4, 128, 32768),
  1, &add_1_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  add_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 128, 1), AI_STRIDE_INIT(4, 4, 4, 256, 32768),
  1, &add_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  audio_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 512, 1), AI_STRIDE_INIT(4, 4, 4, 4, 2048),
  1, &audio_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 128, 1), AI_STRIDE_INIT(4, 4, 4, 256, 32768),
  1, &conv2d_1_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &conv2d_1_weights_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 256, 1), AI_STRIDE_INIT(4, 4, 4, 128, 32768),
  1, &conv2d_2_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 3, 1), AI_STRIDE_INIT(4, 4, 4, 128, 384),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 1, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 12288),
  1, &conv2d_2_weights_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_3_bias_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 256, 1), AI_STRIDE_INIT(4, 4, 4, 128, 32768),
  1, &conv2d_3_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_scratch0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_3_scratch0_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 4096),
  1, &conv2d_3_weights_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 128, 1), AI_STRIDE_INIT(4, 4, 4, 256, 32768),
  1, &conv2d_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_scratch0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 3, 1), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &conv2d_scratch0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_weights, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 49152),
  1, &conv2d_weights_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  convolution_1_bias, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &convolution_1_bias_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  convolution_1_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 256, 1), AI_STRIDE_INIT(4, 4, 4, 128, 32768),
  1, &convolution_1_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  convolution_1_scratch0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 1), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &convolution_1_scratch0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  convolution_1_upsample_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 255, 1), AI_STRIDE_INIT(4, 4, 4, 256, 65280),
  1, &convolution_1_upsample_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  convolution_1_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 64, 4, 1, 32), AI_STRIDE_INIT(4, 4, 256, 8192, 32768),
  1, &convolution_1_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  convolution_2_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &convolution_2_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  convolution_2_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 512, 1), AI_STRIDE_INIT(4, 4, 4, 4, 2048),
  1, &convolution_2_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  convolution_2_scratch0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 1), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &convolution_2_scratch0_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  convolution_2_upsample_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 511, 1), AI_STRIDE_INIT(4, 4, 4, 128, 65408),
  1, &convolution_2_upsample_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  convolution_2_weights, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 32, 4, 1, 1), AI_STRIDE_INIT(4, 4, 128, 128, 512),
  1, &convolution_2_weights_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  convolution_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &convolution_bias_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  convolution_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 128, 1), AI_STRIDE_INIT(4, 4, 4, 256, 32768),
  1, &convolution_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  convolution_scratch0, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 1), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &convolution_scratch0_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  convolution_upsample_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 127, 1), AI_STRIDE_INIT(4, 4, 4, 128, 16256),
  1, &convolution_upsample_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  convolution_weights, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 32, 4, 1, 64), AI_STRIDE_INIT(4, 4, 128, 8192, 32768),
  1, &convolution_weights_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  latent_Transpose_output, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 64, 1), AI_STRIDE_INIT(4, 4, 4, 128, 8192),
  1, &latent_Transpose_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  latent_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 32), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &latent_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_output, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 128, 1), AI_STRIDE_INIT(4, 4, 4, 256, 32768),
  1, &relu_1_output_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 256, 1), AI_STRIDE_INIT(4, 4, 4, 128, 32768),
  1, &relu_2_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  relu_3_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 256, 1), AI_STRIDE_INIT(4, 4, 4, 128, 32768),
  1, &relu_3_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  relu_output, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 128, 1), AI_STRIDE_INIT(4, 4, 4, 256, 32768),
  1, &relu_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  audio_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &audio_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  audio_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_tanh,
  &audio_chain,
  NULL, &audio_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  convolution_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_2_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &convolution_2_weights, &convolution_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &convolution_2_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  convolution_2_layer, 13,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &convolution_2_chain,
  NULL, &audio_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float convolution_2_upsample_scales_data[] = { 1, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    convolution_2_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    convolution_2_upsample_scales_data, convolution_2_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  convolution_2_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_2_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  convolution_2_upsample_layer, 13,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros,
  &convolution_2_upsample_chain,
  NULL, &convolution_2_layer, AI_STATIC, 
  .scales = &convolution_2_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_3_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &relu_3_chain,
  NULL, &convolution_2_upsample_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &convolution_1_output, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_1_layer, 11,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &add_1_chain,
  NULL, &relu_3_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_3_weights, &conv2d_3_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_3_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_layer, 10,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_3_chain,
  NULL, &add_1_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_2_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &relu_2_chain,
  NULL, &conv2d_3_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_2_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_2_chain,
  NULL, &relu_2_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  convolution_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_1_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &convolution_1_weights, &convolution_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &convolution_1_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  convolution_1_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &convolution_1_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float convolution_1_upsample_scales_data[] = { 1, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    convolution_1_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    convolution_1_upsample_scales_data, convolution_1_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  convolution_1_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_1_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  convolution_1_upsample_layer, 7,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros,
  &convolution_1_upsample_chain,
  NULL, &convolution_1_layer, AI_STATIC, 
  .scales = &convolution_1_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_1_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &relu_1_chain,
  NULL, &convolution_1_upsample_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &convolution_output, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_layer, 5,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &add_chain,
  NULL, &relu_1_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_1_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_1_chain,
  NULL, &add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &relu_chain,
  NULL, &conv2d_1_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_weights, &conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_chain,
  NULL, &relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  convolution_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &convolution_weights, &convolution_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &convolution_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  convolution_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &convolution_chain,
  NULL, &conv2d_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float convolution_upsample_scales_data[] = { 1, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    convolution_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    convolution_upsample_scales_data, convolution_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  convolution_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &latent_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &convolution_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  convolution_upsample_layer, 1,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros,
  &convolution_upsample_chain,
  NULL, &convolution_layer, AI_STATIC, 
  .scales = &convolution_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  latent_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &latent_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &latent_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  latent_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &latent_Transpose_chain,
  NULL, &convolution_upsample_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 149124, 1, 1),
    149124, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 99072, 1, 1),
    99072, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_DECODER_IN_NUM, &latent_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_DECODER_OUT_NUM, &audio_output),
  &latent_Transpose_layer, 0x52d22580, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 149124, 1, 1),
      149124, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 99072, 1, 1),
      99072, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_DECODER_IN_NUM, &latent_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_DECODER_OUT_NUM, &audio_output),
  &latent_Transpose_layer, 0x52d22580, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool decoder_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_decoder_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    latent_output_array.data = AI_PTR(g_decoder_activations_map[0] + 66304);
    latent_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 66304);
    latent_Transpose_output_array.data = AI_PTR(g_decoder_activations_map[0] + 58112);
    latent_Transpose_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 58112);
    convolution_upsample_output_array.data = AI_PTR(g_decoder_activations_map[0] + 66304);
    convolution_upsample_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 66304);
    convolution_scratch0_array.data = AI_PTR(g_decoder_activations_map[0] + 82560);
    convolution_scratch0_array.data_start = AI_PTR(g_decoder_activations_map[0] + 82560);
    convolution_output_array.data = AI_PTR(g_decoder_activations_map[0] + 33536);
    convolution_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 33536);
    conv2d_scratch0_array.data = AI_PTR(g_decoder_activations_map[0] + 32768);
    conv2d_scratch0_array.data_start = AI_PTR(g_decoder_activations_map[0] + 32768);
    conv2d_output_array.data = AI_PTR(g_decoder_activations_map[0] + 66304);
    conv2d_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 66304);
    relu_output_array.data = AI_PTR(g_decoder_activations_map[0] + 66304);
    relu_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 66304);
    conv2d_1_scratch0_array.data = AI_PTR(g_decoder_activations_map[0] + 33280);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_decoder_activations_map[0] + 33280);
    conv2d_1_output_array.data = AI_PTR(g_decoder_activations_map[0] + 512);
    conv2d_1_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 512);
    add_output_array.data = AI_PTR(g_decoder_activations_map[0] + 512);
    add_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 512);
    relu_1_output_array.data = AI_PTR(g_decoder_activations_map[0] + 512);
    relu_1_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 512);
    convolution_1_upsample_output_array.data = AI_PTR(g_decoder_activations_map[0] + 33792);
    convolution_1_upsample_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 33792);
    convolution_1_scratch0_array.data = AI_PTR(g_decoder_activations_map[0] + 32768);
    convolution_1_scratch0_array.data_start = AI_PTR(g_decoder_activations_map[0] + 32768);
    convolution_1_output_array.data = AI_PTR(g_decoder_activations_map[0] + 0);
    convolution_1_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 0);
    conv2d_2_scratch0_array.data = AI_PTR(g_decoder_activations_map[0] + 32768);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_decoder_activations_map[0] + 32768);
    conv2d_2_output_array.data = AI_PTR(g_decoder_activations_map[0] + 33152);
    conv2d_2_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 33152);
    relu_2_output_array.data = AI_PTR(g_decoder_activations_map[0] + 65920);
    relu_2_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 65920);
    conv2d_3_scratch0_array.data = AI_PTR(g_decoder_activations_map[0] + 32768);
    conv2d_3_scratch0_array.data_start = AI_PTR(g_decoder_activations_map[0] + 32768);
    conv2d_3_output_array.data = AI_PTR(g_decoder_activations_map[0] + 32896);
    conv2d_3_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 32896);
    add_1_output_array.data = AI_PTR(g_decoder_activations_map[0] + 65664);
    add_1_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 65664);
    relu_3_output_array.data = AI_PTR(g_decoder_activations_map[0] + 0);
    relu_3_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 0);
    convolution_2_upsample_output_array.data = AI_PTR(g_decoder_activations_map[0] + 32768);
    convolution_2_upsample_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 32768);
    convolution_2_scratch0_array.data = AI_PTR(g_decoder_activations_map[0] + 0);
    convolution_2_scratch0_array.data_start = AI_PTR(g_decoder_activations_map[0] + 0);
    convolution_2_output_array.data = AI_PTR(g_decoder_activations_map[0] + 512);
    convolution_2_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 512);
    audio_output_array.data = AI_PTR(g_decoder_activations_map[0] + 2560);
    audio_output_array.data_start = AI_PTR(g_decoder_activations_map[0] + 2560);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool decoder_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_decoder_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    convolution_weights_array.format |= AI_FMT_FLAG_CONST;
    convolution_weights_array.data = AI_PTR(g_decoder_weights_map[0] + 0);
    convolution_weights_array.data_start = AI_PTR(g_decoder_weights_map[0] + 0);
    convolution_bias_array.format |= AI_FMT_FLAG_CONST;
    convolution_bias_array.data = AI_PTR(g_decoder_weights_map[0] + 32768);
    convolution_bias_array.data_start = AI_PTR(g_decoder_weights_map[0] + 32768);
    conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_weights_array.data = AI_PTR(g_decoder_weights_map[0] + 33024);
    conv2d_weights_array.data_start = AI_PTR(g_decoder_weights_map[0] + 33024);
    conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_bias_array.data = AI_PTR(g_decoder_weights_map[0] + 82176);
    conv2d_bias_array.data_start = AI_PTR(g_decoder_weights_map[0] + 82176);
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_decoder_weights_map[0] + 82432);
    conv2d_1_weights_array.data_start = AI_PTR(g_decoder_weights_map[0] + 82432);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_decoder_weights_map[0] + 98816);
    conv2d_1_bias_array.data_start = AI_PTR(g_decoder_weights_map[0] + 98816);
    convolution_1_weights_array.format |= AI_FMT_FLAG_CONST;
    convolution_1_weights_array.data = AI_PTR(g_decoder_weights_map[0] + 99072);
    convolution_1_weights_array.data_start = AI_PTR(g_decoder_weights_map[0] + 99072);
    convolution_1_bias_array.format |= AI_FMT_FLAG_CONST;
    convolution_1_bias_array.data = AI_PTR(g_decoder_weights_map[0] + 131840);
    convolution_1_bias_array.data_start = AI_PTR(g_decoder_weights_map[0] + 131840);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_decoder_weights_map[0] + 131968);
    conv2d_2_weights_array.data_start = AI_PTR(g_decoder_weights_map[0] + 131968);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_decoder_weights_map[0] + 144256);
    conv2d_2_bias_array.data_start = AI_PTR(g_decoder_weights_map[0] + 144256);
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(g_decoder_weights_map[0] + 144384);
    conv2d_3_weights_array.data_start = AI_PTR(g_decoder_weights_map[0] + 144384);
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(g_decoder_weights_map[0] + 148480);
    conv2d_3_bias_array.data_start = AI_PTR(g_decoder_weights_map[0] + 148480);
    convolution_2_weights_array.format |= AI_FMT_FLAG_CONST;
    convolution_2_weights_array.data = AI_PTR(g_decoder_weights_map[0] + 148608);
    convolution_2_weights_array.data_start = AI_PTR(g_decoder_weights_map[0] + 148608);
    convolution_2_bias_array.format |= AI_FMT_FLAG_CONST;
    convolution_2_bias_array.data = AI_PTR(g_decoder_weights_map[0] + 149120);
    convolution_2_bias_array.data_start = AI_PTR(g_decoder_weights_map[0] + 149120);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_decoder_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_DECODER_MODEL_NAME,
      .model_signature   = AI_DECODER_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 6412577,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x52d22580,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_decoder_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_DECODER_MODEL_NAME,
      .model_signature   = AI_DECODER_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 6412577,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x52d22580,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_decoder_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_decoder_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_decoder_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_decoder_create(network, AI_DECODER_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_decoder_data_params_get(&params) != true) {
    err = ai_decoder_get_error(*network);
    return err;
  }
#if defined(AI_DECODER_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_DECODER_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_decoder_init(*network, &params) != true) {
    err = ai_decoder_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_decoder_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_decoder_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_decoder_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_decoder_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= decoder_configure_weights(net_ctx, params);
  ok &= decoder_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_decoder_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_decoder_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_DECODER_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

