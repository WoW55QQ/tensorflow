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
using UnityEngine;
using UnityEngine.Video;

public class HandTracking : MonoBehaviour 
{
    [Tooltip("Configurable TFLite model.")]
    public int InputW = 256;
    public int InputH = 256;
    public TextAsset PalmDetection;
    public TextAsset HandLandmark3D;
    public int PalmDetectionLerpFrameCount = 3;
    public int HandLandmark3DLerpFrameCount = 4;
    public bool UseGPU = true;
    private RenderTexture videoTexture;
    private Texture2D texture;

    private Inferencer inferencer = new Inferencer();
    private GameObject debugPlane;
    private DebugRenderer debugRenderer;

    void Awake() { QualitySettings.vSyncCount = 0; }

    void Start() 
    {
        InitTexture();
        inferencer.Init(PalmDetection, HandLandmark3D, UseGPU, 
                        PalmDetectionLerpFrameCount, HandLandmark3DLerpFrameCount);
        debugPlane = GameObject.Find("TensorFlowLite");
        debugRenderer = debugPlane.GetComponent<DebugRenderer>();
        debugRenderer.Init(inferencer.InputWidth, inferencer.InputHeight, debugPlane);
    }
    private void InitTexture()
    { 
        var rectTransform = GetComponent<RectTransform>();
        var renderer = GetComponent<Renderer>();

        var videoPlayer = GetComponent<VideoPlayer>();
        int width = (int)rectTransform.rect.width;
        int height = (int)rectTransform.rect.height;
        videoTexture = new RenderTexture(width, height, 24);
        videoPlayer.targetTexture = videoTexture;
        renderer.material.mainTexture = videoTexture;
        videoPlayer.Play();

        texture = new Texture2D(videoTexture.width, videoTexture.height, TextureFormat.RGB24, false);
     }

    void Update() 
    {
        Graphics.SetRenderTarget(videoTexture);
        texture.ReadPixels(new Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
        texture.Apply();
        Graphics.SetRenderTarget(null);

        inferencer.Update(texture);
    }

    public void OnRenderObject() 
    {
        if (!inferencer.Initialized){ return; }

        bool debugHandLandmarks3D = true;
        if (debugHandLandmarks3D)
        { 
            var handLandmarks = inferencer.HandLandmarks;
            debugRenderer.DrawHand3D(handLandmarks);
        }
    }

    void OnDestroy(){ inferencer.Destroy(); }
}
