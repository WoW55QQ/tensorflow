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
    public TextAsset PalmDetection;
    public TextAsset HandLandmark3D;

    private RenderTexture videoTexture;
    private Texture2D texture;
    private float videoWidth, videoHeight;
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

        texture = new Texture2D(videoTexture.width, videoTexture.height);
        videoWidth = texture.width;
        videoHeight = texture.height;
     }

    void Update() 
    {
        UpdateTexture();

        if (isPosing) { return; }
        isPosing = true;
        StartCoroutine("UpdateInferencer", texture);
    }

    private void UpdateTexture()
    { 
        Graphics.SetRenderTarget(videoTexture);
        texture.ReadPixels(new Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
        texture.Apply();
        Graphics.SetRenderTarget(null);
    }

    IEnumerator UpdateInferencer(Texture2D texture) 
    {
        int width = inferencer.InputWidth;
        int height = inferencer.InputHeight;
        Texture2D resizedTexture = ResizeTexture(texture, width, height);

        inferencer.Update(resizedTexture);

        Destroy(resizedTexture);

        isPosing = false;
        yield return null;
    }

    private Texture2D ResizeTexture(Texture2D src, int width, int height) 
    {
        float videoShortSide = (videoWidth > videoHeight) ? videoHeight : videoWidth;
        float aspectWidth = videoWidth / videoShortSide;
        float aspectHeight = videoHeight / videoShortSide;

        src.filterMode = FilterMode.Trilinear;
        src.Apply(true);

        RenderTexture rt = new RenderTexture(width, height, 32);
        Graphics.SetRenderTarget(rt);
        GL.LoadPixelMatrix(0, aspectWidth, 0, aspectHeight);
        //RotateTexture();
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, aspectWidth, aspectHeight), src);

        Rect rect = new Rect(0, 0, width, height);
        Texture2D dst = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.ARGB32, true);
        dst.ReadPixels(rect, 0, 0, true);
        Graphics.SetRenderTarget(null);
        Destroy(rt);

        return dst;
    }

    private void RotateTexture()
    {
        Vector3 t = new Vector3(0, 1, 0);
        Quaternion r = Quaternion.Euler(0, 0, -90);
        Vector3 s = Vector3.one;
        Matrix4x4 m = Matrix4x4.identity;
        m.SetTRS(t, r, s);
        GL.MultMatrix(m);
    }

    public void OnRenderObject() {
        if (!inferencer.Initialized){ return; }

        //var inputs = inferencer.Inputs;
        //renderer.DrawInput(inputs);

        var palmBox = inferencer.PalmBox;
        var palmKeypoints = inferencer.PalmKeypoints;
        //Debug.Log(string.Format("Palm box {0:0.0}, {1:0.0}, {2:0.0}, {3:0.0}", palmBox.x, palmBox.y, palmBox.width, palmBox.height));
        renderer.DrawPalm(palmBox, palmKeypoints);
    }

    void OnDestroy() { inferencer.Destroy(); }
}
