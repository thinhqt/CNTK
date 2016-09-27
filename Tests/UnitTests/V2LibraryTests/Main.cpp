//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"
#include <functional>

using namespace CNTK;

void NDArrayViewTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();
void TrainerTests();
void TestCifarResnet();
void FunctionTests();
void TrainLSTMSequenceClassifer();
void SerializationTests();
void LearnerTests();
void TrainSequenceToSequenceTranslator();
void MultiThreadsEvaluationWithNewFunction(const DeviceDescriptor&, const int);
void MultiThreadsEvaluationWithClone(const DeviceDescriptor&, const int);

int main()
{
    NDArrayViewTests();
    TensorTests();
    FunctionTests();

    FeedForwardTests();
    RecurrentFunctionTests();

    TrainerTests();
    SerializationTests();
    LearnerTests();

    TestCifarResnet();
    TrainLSTMSequenceClassifer();

    TrainSequenceToSequenceTranslator();

    // Test multi-threads evaluation with new function
    fprintf(stderr, "Test multi-threaded evaluation with new function on CPU.\n");
    MultiThreadsEvaluationWithNewFunction(DeviceDescriptor::CPUDevice(), 2);
#ifndef CPUONLY
    fprintf(stderr, "Test multi-threaded evaluation with new function on GPU\n");
    MultiThreadsEvaluationWithNewFunction(DeviceDescriptor::GPUDevice(0), 2);
#endif

    //// Test multi-threads evaluation using clone.
    fprintf(stderr, "Test multi-threaded evaluation using clone on CPU.\n");
    MultiThreadsEvaluationWithClone(DeviceDescriptor::CPUDevice(), 2);
#ifndef CPUONLY
    fprintf(stderr, "Test multi-threaded evaluation using clone on GPU.\n");
    MultiThreadsEvaluationWithClone(DeviceDescriptor::GPUDevice(0), 2);
#endif

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
