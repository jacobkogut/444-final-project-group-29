/*
 * vq_pipeline.h
 *
 *  Created on: Nov 26, 2025
 *      Author: williamzimmerman
 */

#ifndef INC_VQ_PIPELINE_H_
#define INC_VQ_PIPELINE_H_


#include <stdint.h>

// Initialize the AI models
void VQ_Init(void);

// Run the full process: Audio In -> Encoder -> VQ -> Decoder -> Audio Out
// input: array of 512 floats
// output: array of 512 floats
void VQ_Process(float *input_buffer, float *output_buffer);

void MuLaw_Encode_Buffer(float* input, float* output, int length);

void MuLaw_Decode_Buffer(float* input, float* output, int length);

// --- TRANSMITTER SIDE ---
// Takes Raw Audio -> MuLaw -> Encoder -> VQ Search -> Output Indices
// input_audio: 512 floats (Raw Mic)
// output_indices: 64 integers (To be transmitted)
void Tx_Process(float *input_audio, uint16_t *output_indices);

// --- RECEIVER SIDE ---
// Takes Indices -> Codebook Lookup -> Decoder -> Inverse MuLaw -> Output Audio
// input_indices: 64 integers (Received)
// output_audio: 512 floats (To Speaker)
void Rx_Process(uint16_t *input_indices, float *output_audio);


#endif /* INC_VQ_PIPELINE_H_ */
