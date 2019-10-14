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
using System.Collections;
using UnityEngine;
using UnityEngine.Video;

public class HandTracking : MonoBehaviour 
{
    [Tooltip("Configurable TFLite model.")]
    public int InputW = 256;
    public int InputH = 256;
    public TextAsset PalmDetection;
    public TextAsset HandLandmark3D;

    private RenderTexture videoTexture;
    private Texture2D texture;
    private bool isPosing = false;

    private Inferencer inferencer = new Inferencer();
    private DebugRenderer renderer;

    void Awake() { QualitySettings.vSyncCount = 0; }

    void Start() 
    {
        InitTexture();
        inferencer.Init(PalmDetection, HandLandmark3D);
        renderer = GameObject.Find("TensorFlowLite").GetComponent<DebugRenderer>();
        renderer.Init(inferencer.InputWidth, inferencer.InputHeight);
    }
    private void InitTexture()
    { 
        RectTransform rectTransform = GetComponent<RectTransform>();
        Renderer renderer = GetComponent<Renderer>();

        VideoPlayer videoPlayer = GetComponent<VideoPlayer>();
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
        if (isPosing) { return; }
        isPosing = true;
        StartCoroutine("UpdateInferencer", texture);
    }

    IEnumerator UpdateInferencer(Texture2D texture) 
    {
        Graphics.SetRenderTarget(videoTexture);
        texture.ReadPixels(new Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
        texture.Apply();
        Graphics.SetRenderTarget(null);

        inferencer.Update(texture);

        isPosing = false;
        yield return null;
    }

    public void OnRenderObject() 
    {
        if (!inferencer.Initialized){ return; }

        var palmDetectionInputs = inferencer.PalmDetectionInputs;
        //renderer.DrawInput(palmDetectionInputs);

        var palmBox = inferencer.PalmBox;
        var palmKeypoints = inferencer.PalmKeypoints;
        var handBox = inferencer.HandBox;
        var handCenter = inferencer.HandCenter;
        //Debug.Log(string.Format("Palm box {0:0.0}, {1:0.0}, {2:0.0}, {3:0.0}", palmBox.x, palmBox.y, palmBox.width, palmBox.height));
        //renderer.DrawPalm(palmBox, palmKeypoints, handBox, handCenter);

        var handLandmarksInputs = inferencer.HandLandmarksInputs;
        //renderer.DrawInput(handLandmarksInputs);

        var handLandmarks = inferencer.HandLandmarks;
        renderer.DrawHand(handLandmarks);
    }

    void OnDestroy(){ inferencer.Destroy(); }
}
