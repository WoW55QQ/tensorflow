/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
using System;
using System.Runtime.InteropServices;

using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteDelegate = System.IntPtr;
using TfLiteModel = System.IntPtr;
using TfLiteTensor = System.IntPtr;
using System.Text;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Simple C# bindings for the experimental TensorFlowLite C API.
    /// </summary>
    public class Interpreter : IDisposable
    {
        private const string TensorFlowLibrary = "tensorflowlite_c";

        private TfLiteModel model;
        private TfLiteInterpreter interpreter;
        private TfLiteInterpreterOptions interpreterOptions;
        private const int numThreads = 4;

        public Interpreter(byte[] modelData) 
        {
            GCHandle modelDataHandle = GCHandle.Alloc(modelData, GCHandleType.Pinned);
            IntPtr modelDataPtr = modelDataHandle.AddrOfPinnedObject();

            model = TfLiteModelCreate(modelDataPtr, modelData.Length);
            if (model == IntPtr.Zero){ throw new Exception("Failed to create TensorFlowLite Model"); }
            
            interpreterOptions = TfLiteInterpreterOptionsCreate();
            TfLiteInterpreterOptionsSetNumThreads(interpreterOptions, numThreads);

            TFLInterpreterErrorReporter reporter = InterpreterErrorReporter;
            TfLiteInterpreterOptionsSetErrorReporter(interpreterOptions, reporter, interpreter);

            interpreter = TfLiteInterpreterCreate(model, interpreterOptions);
            if (interpreter == IntPtr.Zero){ throw new Exception("Failed to create TensorFlowLite Interpreter"); }
        }

        ~Interpreter() { Dispose(); }

        public void Dispose() 
        {
            if (interpreterOptions != IntPtr.Zero){ TfLiteInterpreterOptionsDelete(interpreterOptions); }
            interpreterOptions = IntPtr.Zero;

            if (interpreter != IntPtr.Zero){ TfLiteInterpreterDelete(interpreter); }
            interpreter = IntPtr.Zero;

            if (model != IntPtr.Zero){ TfLiteModelDelete(model); }
            model = IntPtr.Zero;
        }

        public void ResizeInputTensor(int inputTensorIndex, int[] inputTensorShape) 
        {
            ThrowIfError(TfLiteInterpreterResizeInputTensor(interpreter, inputTensorIndex, inputTensorShape, inputTensorShape.Length));
        }
        public void AllocateTensors() { ThrowIfError(TfLiteInterpreterAllocateTensors(interpreter)); }

        public int GetInputTensorCount() { return TfLiteInterpreterGetInputTensorCount(interpreter); }

        public TfLiteTensor GetInputTensor(int inputIndex) { return TfLiteInterpreterGetInputTensor(interpreter, inputIndex); }

        public int GetOutputTensorCount() { return TfLiteInterpreterGetOutputTensorCount(interpreter); }

        public TfLiteTensor GetOutputTensor(int outputIndex) { return TfLiteInterpreterGetOutputTensor(interpreter, outputIndex); }

        public TfLiteType GetTensorType(TfLiteTensor tensor) { return TfLiteTensorType(tensor); }

        public int GetTensorNumDims(TfLiteTensor tensor) { return TfLiteTensorNumDims(tensor); }

        public int GetTensorDim(TfLiteTensor tensor, int dimIndex) { return TfLiteTensorDim(tensor, dimIndex); }

        public int GetTensorByteSize(TfLiteTensor tensor) { return TfLiteTensorByteSize(tensor); }

        public IntPtr GetTensorData(TfLiteTensor tensor) { return TfLiteTensorData(tensor); }

        public string GetTensorName(TfLiteTensor tensor) { 
            byte[] str = new byte[0xFF];
            int count = vsprintf(str, TfLiteTensorName(tensor), IntPtr.Zero);
            return Encoding.ASCII.GetString(str, 0, count);
        }

        public TfLiteQuantizationParams GetTensorQuantizationParams(TfLiteTensor tensor) { return TfLiteTensorQuantizationParams(tensor); }

        public void SetInputTensorData(int inputTensorIndex, Array inputTensorData) 
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(inputTensorData, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
            ThrowIfError(TfLiteTensorCopyFromBuffer(tensor, tensorDataPtr, Buffer.ByteLength(inputTensorData)));
        }
        public void Invoke(){ ThrowIfError(TfLiteInterpreterInvoke(interpreter)); }

        public void GetOutputTensorData(int outputTensorIndex, Array outputTensorData) 
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(outputTensorData, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
            ThrowIfError(TfLiteTensorCopyToBuffer(tensor, tensorDataPtr, Buffer.ByteLength(outputTensorData)));
        }

        private static unsafe void ThrowIfError(TfLiteStatus resultCode) 
        {
            if (resultCode != 0)
            {
                throw new Exception("TensorFlowLite operation failed.");
            }
        }

#region Externs
        public enum TfLiteStatus { kTfLiteOk = 0, kTfLiteError = 1 };
        public enum TfLiteType 
        {
            kTfLiteNoType = 0, kTfLiteFloat32 = 1, kTfLiteInt32 = 2, kTfLiteUInt8 = 3, kTfLiteInt64 = 4,
            kTfLiteString = 5, kTfLiteBool = 6, kTfLiteInt16 = 7, kTfLiteComplex64 = 8, kTfLiteInt8 = 9, kTfLiteFloat16 = 10,
        };
        public struct TfLiteQuantizationParams 
        {
            float scale;
            Int32 zero_point;
        };

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void TFLInterpreterErrorReporter(IntPtr userData, string format, IntPtr args);
        private static void InterpreterErrorReporter(IntPtr userData, string format, IntPtr args)
        {
            byte[] str = new byte[0xFF];
            int count = vsprintf(str, format, args);
            Debug.LogError(Encoding.ASCII.GetString(str, 0, count));
        }

        [DllImport("msvcrt")]
        private static extern unsafe int vsprintf(byte[] str, string format, IntPtr args);

        [DllImport("msvcrt")]
        private static extern unsafe int vsprintf(byte[] str, IntPtr format, IntPtr args);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelCreate(IntPtr modelData, Int32 modelSize);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelDelete(TfLiteModel model);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreterOptions TfLiteInterpreterOptionsCreate();

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions options);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions options, Int32 numThreads);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions options, TfLiteDelegate delegate_);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsSetErrorReporter(TfLiteInterpreterOptions options, 
                                                                                    TFLInterpreterErrorReporter reporter, IntPtr userData);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteInterpreterCreate(TfLiteModel model, TfLiteInterpreterOptions optionalOptions);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterDelete(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteInterpreterGetInputTensorCount(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetInputTensor(TfLiteInterpreter interpreter, Int32 inputIndex);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteInterpreterResizeInputTensor(TfLiteInterpreter interpreter, Int32 inputIndex, 
                                                                                    Int32[] inputDims, Int32 inputDimsSize);
        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteInterpreterGetOutputTensorCount(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetOutputTensor(TfLiteInterpreter interpreter, Int32 outputIndex);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteType TfLiteTensorType(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteTensorNumDims(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteTensorDim(TfLiteTensor tensor, Int32 dimIndex);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteTensorByteSize(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteTensorData(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteTensorName(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteQuantizationParams TfLiteTensorQuantizationParams(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor tensor, IntPtr inputData, Int32 inputDataSize);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteTensorCopyToBuffer(TfLiteTensor tensor, IntPtr outputData, Int32 outputDataSize);

#endregion
    }
}
