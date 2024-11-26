// softmax.h
#ifndef SOFTMAX_H
#define SOFTMAX_H

void softmax(float *m, unsigned int R, unsigned int C);

void softmax_batched(float *m, unsigned int B, unsigned int NH, unsigned int T);

#endif
