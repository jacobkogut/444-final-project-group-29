/**
  ******************************************************************************
  * @file    encoder_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-11-27T01:23:36-0500
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

#ifndef ENCODER_DATA_PARAMS_H
#define ENCODER_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_ENCODER_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_encoder_data_weights_params[1]))
*/

#define AI_ENCODER_DATA_CONFIG               (NULL)


#define AI_ENCODER_DATA_ACTIVATIONS_SIZES \
  { 98560, }
#define AI_ENCODER_DATA_ACTIVATIONS_SIZE     (98560)
#define AI_ENCODER_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ENCODER_DATA_ACTIVATION_1_SIZE    (98560)



#define AI_ENCODER_DATA_WEIGHTS_SIZES \
  { 149248, }
#define AI_ENCODER_DATA_WEIGHTS_SIZE         (149248)
#define AI_ENCODER_DATA_WEIGHTS_COUNT        (1)
#define AI_ENCODER_DATA_WEIGHT_1_SIZE        (149248)



#define AI_ENCODER_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_encoder_activations_table[1])

extern ai_handle g_encoder_activations_table[1 + 2];



#define AI_ENCODER_DATA_WEIGHTS_TABLE_GET() \
  (&g_encoder_weights_table[1])

extern ai_handle g_encoder_weights_table[1 + 2];


#endif    /* ENCODER_DATA_PARAMS_H */
