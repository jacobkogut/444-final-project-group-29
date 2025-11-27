/**
  ******************************************************************************
  * @file    decoder_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-11-27T01:23:50-0500
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef DECODER_DATA_PARAMS_H
#define DECODER_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_DECODER_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_decoder_data_weights_params[1]))
*/

#define AI_DECODER_DATA_CONFIG               (NULL)


#define AI_DECODER_DATA_ACTIVATIONS_SIZES \
  { 99072, }
#define AI_DECODER_DATA_ACTIVATIONS_SIZE     (99072)
#define AI_DECODER_DATA_ACTIVATIONS_COUNT    (1)
#define AI_DECODER_DATA_ACTIVATION_1_SIZE    (99072)



#define AI_DECODER_DATA_WEIGHTS_SIZES \
  { 149124, }
#define AI_DECODER_DATA_WEIGHTS_SIZE         (149124)
#define AI_DECODER_DATA_WEIGHTS_COUNT        (1)
#define AI_DECODER_DATA_WEIGHT_1_SIZE        (149124)



#define AI_DECODER_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_decoder_activations_table[1])

extern ai_handle g_decoder_activations_table[1 + 2];



#define AI_DECODER_DATA_WEIGHTS_TABLE_GET() \
  (&g_decoder_weights_table[1])

extern ai_handle g_decoder_weights_table[1 + 2];


#endif    /* DECODER_DATA_PARAMS_H */
