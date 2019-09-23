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
using TensorFlowLite;
using UnityEngine;

using static UnityEngine.Mathf;

// https://github.com/google/mediapipe/blob/master/mediapipe/graphs/hand_tracking/hand_detection_gpu.pbtxt
public class Inferencer 
{
    private const int NN_INPUT_WIDTH = 256;
    private const int NN_INPUT_HEIGHT = 256;
    private const int NN_INPUT_CHANNEL = 3;
    private const int NN_NUM_CLASSES = 1;
    private const int NN_NUM_BOXES = 2944;
    private const int NN_NUM_COORDS = 18; // box(x,y,z,w) + keypoints(x,y) * 7
    private const int NN_NUM_KEYPOINTS = 7;

    private const int NN_NUM_KEYPOINTS_3D = 21;

    private Interpreter palmDetectionInterpreter;
    private Interpreter handLandmark3DInterpreter;

    private float[,,,] inputs = new float[1, NN_INPUT_HEIGHT, NN_INPUT_WIDTH, NN_INPUT_CHANNEL];
    private float[,] anchors = new float[NN_NUM_BOXES, 4];
    private float[,,] regressors = new float[1, NN_NUM_BOXES, NN_NUM_COORDS];
    private float[,,] classificators = new float[1, NN_NUM_BOXES, NN_NUM_CLASSES];
    private float[,] keypoints3d = new float[1, NN_NUM_KEYPOINTS_3D * 3];

    public bool Initialized = false;
    public int InputWidth { get { return NN_INPUT_WIDTH; } }
    public int InputHeight { get { return NN_INPUT_HEIGHT; } }
    public float[,,,] Inputs { get { return inputs; } }
    public int PalmNumKeypoints { get { return NN_NUM_KEYPOINTS; } }

    public Rect PalmBox = new Rect();
    public Vector2[] PalmKeypoints = new Vector2[NN_NUM_KEYPOINTS];

    public void Init(TextAsset palmDetection, TextAsset handLandmark3D) 
    {
        palmDetectionInterpreter = InitInterpreter(palmDetection);
        handLandmark3DInterpreter = InitInterpreter(handLandmark3D);
        InitAnchors(anchors, NN_INPUT_WIDTH, NN_INPUT_HEIGHT);
    }
    private Interpreter InitInterpreter(TextAsset model)
    { 
        Interpreter interpreter = new Interpreter(model.bytes);
        interpreter.AllocateTensors();

        var inputTensorCount = interpreter.GetInputTensorCount();
        for(int i = 0; i < inputTensorCount; ++i){ DebugTensorData(interpreter, interpreter.GetInputTensor(i)); }

        var outputTensorCount = interpreter.GetOutputTensorCount();
        for(int i = 0; i < outputTensorCount; ++i){ DebugTensorData(interpreter, interpreter.GetOutputTensor(i)); }

        return interpreter;
    }
    private void DebugTensorData(Interpreter interpreter, IntPtr tensor) 
    { 
        var type = interpreter.GetTensorType(tensor);
        int numDims = interpreter.GetTensorNumDims(tensor);
        int[] dims = new int[numDims];
        for(int i = 0; i < numDims; ++i) { dims[i] = interpreter.GetTensorDim(tensor, i); }
        int byteSize = interpreter.GetTensorByteSize(tensor);
        IntPtr data = interpreter.GetTensorData(tensor);
        var name = interpreter.GetTensorName(tensor);
        Interpreter.TfLiteQuantizationParams tensorQuantizationParams = interpreter.GetTensorQuantizationParams(tensor);
    }

    private void InitAnchors(float[,] anchors, int width, int height)
    {
        const int NN_NUM_LAYERS = 5;
        const float NN_ANCHOR_OFFSET_X = 0.5f;
        const float NN_ANCHOR_OFFSET_Y = 0.5f;
        int[] NN_STRIDES = { 8, 16, 32, 32, 32 };

        int index = 0;
        for(int layer = 0; layer < NN_NUM_LAYERS; ++layer)
        {
            float stride = NN_STRIDES[layer];
            int featureMapHeight = CeilToInt(width / stride);
            int featureMapWidth = CeilToInt(height / stride);
            for(int y = 0; y < featureMapHeight; ++y)
            {
                for(int x = 0; x < featureMapWidth; ++x)
                {
                    for(int aspectRatios = 0; aspectRatios < 2; ++aspectRatios)
                    { 
                        anchors[index, 0] = (x + NN_ANCHOR_OFFSET_X) / featureMapWidth;
                        anchors[index, 1] = (y + NN_ANCHOR_OFFSET_Y) / featureMapHeight;
                        anchors[index, 2] = 1.0f;
                        anchors[index, 3] = 1.0f;
                        ++index;
                    }
                }
            }
        }
    }

    public void Update(Texture2D texture) 
    {
        UpdateShapes(texture);
        UpdatePalmDetection();

        Initialized = true;
    }
    
    private unsafe void UpdateShapes(Texture2D texture) 
    {
        const int ARGB = 4, A = 1;
        const float ItoF = 1.0f / 255.0f;

        byte[] pixels = texture.GetRawTextureData();
        
        Array.Clear(inputs, 0, inputs.Length);
        fixed (byte* src = pixels) 
        {
            for (int y = 0; y < NN_INPUT_HEIGHT; ++y) 
            {
                int srcHeight = y * NN_INPUT_WIDTH;
                for (int x = 0; x < NN_INPUT_WIDTH; ++x) 
                {
                    int flipedX = (NN_INPUT_WIDTH - 1) - x;
                    int srcWidth = flipedX;
                    byte* srcPos = src + (srcHeight + srcWidth) * ARGB + A;

                    byte r = *(srcPos++);
                    byte g = *(srcPos++);
                    byte b = *(srcPos++);
                    inputs[0, y, x, 0] = r * ItoF;
                    inputs[0, y, x, 1] = g * ItoF;
                    inputs[0, y, x, 2] = b * ItoF;
                }
            }
        }
    }

    private void UpdatePalmDetection() 
    { 
        const int NN_BOX_COORD_OFFSET = 0;
        const int NN_KEYPOINT_COORD_OFFSET = 4;
        const int NN_NUM_VALUES_PERKEYPOINT = 2;
        const float NN_SCORE_CLIPPING_THRESH = 100.0f;
        const float NN_MIN_SCORE_THRESH = 0.7f;
        const float NN_X_SCALE = 256.0f;
        const float NN_Y_SCALE = 256.0f;
        const float NN_H_SCALE = 256.0f;
        const float NN_W_SCALE = 256.0f;

        palmDetectionInterpreter.SetInputTensorData(0, inputs);

        //float startTimeSeconds = Time.realtimeSinceStartup;
        palmDetectionInterpreter.Invoke();
        //float inferenceTimeSeconds = (Time.realtimeSinceStartup - startTimeSeconds) * 10000;
        //Debug.Log(string.Format("Palm detection {0:0.0000} ms", inferenceTimeSeconds));

        Array.Clear(regressors, 0, regressors.Length);
        palmDetectionInterpreter.GetOutputTensorData(0, regressors);
        Array.Clear(classificators, 0, classificators.Length);
        palmDetectionInterpreter.GetOutputTensorData(1, classificators);

        float maxScore = -1.0f;
        for(int i = 0; i < NN_NUM_BOXES; ++i)
        {
            float score = classificators[0, i, 0];
            score = Clamp(score, -NN_SCORE_CLIPPING_THRESH, NN_SCORE_CLIPPING_THRESH);
            score = Sigmoid(score);
            if(score < NN_MIN_SCORE_THRESH){ continue; }
            if(score < maxScore){ continue; }
            maxScore = score;

            float centerX = regressors[0, i, NN_BOX_COORD_OFFSET + 0];
            float centerY = regressors[0, i, NN_BOX_COORD_OFFSET + 1];
            float w = regressors[0, i, NN_BOX_COORD_OFFSET + 2];
            float h = regressors[0, i, NN_BOX_COORD_OFFSET + 3];

            centerX = centerX / NN_X_SCALE * anchors[i, 2] + anchors[i, 0];
            centerY = centerY / NN_Y_SCALE * anchors[i, 3] + anchors[i, 1];
            w = w / NN_W_SCALE * anchors[i, 2];
            h = h / NN_H_SCALE * anchors[i, 3];

            float boxMinY = centerY - h / 2.0f;
            float boxMinX = centerX - w / 2.0f;
            float boxMaxY = centerY + h / 2.0f;
            float boxMaxX = centerX + w / 2.0f;
            PalmBox.Set(boxMinX, boxMinY, boxMaxX - boxMinX, boxMaxY - boxMinY);

            int jMax = NN_NUM_KEYPOINTS;
            for(int j = 0; j < jMax; ++j)
            { 
                int ofset = NN_BOX_COORD_OFFSET + NN_KEYPOINT_COORD_OFFSET + j * NN_NUM_VALUES_PERKEYPOINT;
                float keypointX = regressors[0, i, ofset + 0];
                float keypointY = regressors[0, i, ofset + 1];
                keypointX = keypointX / NN_X_SCALE * anchors[i, 2] + anchors[i, 0];
                keypointY = keypointY / NN_Y_SCALE * anchors[i, 3] + anchors[i, 1];
                PalmKeypoints[j].Set(keypointX, keypointY);
            }
        }
    }

    private float Sigmoid(float x){ return 1.0f / (1.0f + Exp(-x)); }
 
    public void Destroy() 
    { 
        palmDetectionInterpreter.Dispose(); 
        handLandmark3DInterpreter.Dispose();
    }
}
