using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using static UnityEngine.Mathf;

public class DebugRenderer : MonoBehaviour {
    private float planeScale = 0.039f;
    private float planePos = -5.0f;
    private const float circleRad = 0.1f;

    private int width = 0, height = 0;
    private Material lineMaterial;

    public void Init(int width, int height) 
    { 
        this.width = width; 
        this.height = height; 
    }

    private void CreateLineMaterial() 
    {
        if (lineMaterial) { return; }

        Shader shader = Shader.Find("Hidden/Internal-Colored");
        lineMaterial = new Material(shader);
        lineMaterial.hideFlags = HideFlags.HideAndDontSave;
        lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        lineMaterial.SetInt("_ZWrite", 0);
    }

    private float FitPlane(float pos) { return pos * planeScale + planePos; }

    private float AdjScale(float pos, float src, float dst) 
    {
        float scale = src / dst;
        float center = dst / 2.0f;
        return (pos - center) * scale + center; 
    }

    private void DrawBegin() 
    {
        CreateLineMaterial();
        lineMaterial.SetPass(0);
        GL.PushMatrix();
        GL.MultMatrix(transform.localToWorldMatrix);
    }
    private void DrawEnd() { GL.PopMatrix(); }

    private void DrawCircle(float x, float y, float theta)
    { 
        float posX = FitPlane(x) + Cos(theta) * circleRad;
        float posY = FitPlane(y) + Sin(theta) * circleRad;
        GL.Vertex3(posX, 0.0f, posY);
    }

    public void DrawInput(float[,,,] inputs) 
    {
        Color color = Color.black;
        Color padding = Color.gray;
        Color cleared = Color.black;

        DrawBegin();
        for (int y = 0; y < height; ++y) 
        {
            GL.Begin(GL.LINE_STRIP);
            for (int x = 0; x < width; ++x) 
            {
                color.r = inputs[0, y, x, 0] * 0.5f + 0.5f;
                color.g = inputs[0, y, x, 1] * 0.5f + 0.5f;
                color.b = inputs[0, y, x, 2] * 0.5f + 0.5f;
                color.a = 1.0f;

//                if(color == padding) { continue; }
//                if(color == cleared) { continue; }

                GL.Color(color);
                GL.Vertex3(FitPlane(x), 0.0f, FitPlane(y));
            }
            GL.End();
        }
        DrawEnd();
    }

    public void DrawPalm(Rect box, Vector2[] keypoints, Vector2[] handBox, Vector2 handCenter) 
    {
        DrawBegin();

        // Palm rect
        var xMin = box.x * width;
        var xMax = (box.x + box.width) * width;
        var yMin = box.y * height;
        var yMax = (box.y + box.height) * height;
        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.green);
        GL.Vertex3(FitPlane(xMin), 0.0f, FitPlane(yMin));
        GL.Vertex3(FitPlane(xMin), 0.0f, FitPlane(yMax));
        GL.Vertex3(FitPlane(xMax), 0.0f, FitPlane(yMax));
        GL.Vertex3(FitPlane(xMax), 0.0f, FitPlane(yMin));
        GL.Vertex3(FitPlane(xMin), 0.0f, FitPlane(yMin));
        GL.End();

        // Palm keypoints
        for(int i = 0; i < keypoints.Length; ++i)
        { 
            GL.Begin(GL.LINES);
            GL.Color(Color.blue);
            for (float theta = 0.0f; theta < (2.0f * PI); theta += 0.01f) 
            {
                float x = keypoints[i].x * width;
                float y = keypoints[i].y * height;
                DrawCircle(x, y, theta);
            }
            GL.End();
        }

        float scaleX = 0.0f;

        // Hand rect center
        GL.Begin(GL.LINES);
        GL.Color(Color.yellow);
        for (float theta = 0.0f; theta < (2.0f * PI); theta += 0.01f) 
        {
            DrawCircle(handCenter.x + scaleX, handCenter.y, theta);
        }
        GL.End();

        float srcW = 256.0f * 1.25f, dstW = 256.0f;
        float srcH = 256.0f * 1.25f, dstH = 256.0f;

        // Hand rect
        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.red);
        GL.Vertex3(FitPlane(handBox[0].x), 0.0f, FitPlane(handBox[0].y));
        GL.Vertex3(FitPlane(handBox[1].x), 0.0f, FitPlane(handBox[1].y));
        GL.Vertex3(FitPlane(handBox[2].x), 0.0f, FitPlane(handBox[2].y));
        GL.Vertex3(FitPlane(handBox[3].x), 0.0f, FitPlane(handBox[3].y));
        GL.Vertex3(FitPlane(handBox[0].x), 0.0f, FitPlane(handBox[0].y));

        GL.End();

        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.blue);
        GL.Vertex3(FitPlane(0), 0.0f, FitPlane(0));
        GL.Vertex3(FitPlane(0), 0.0f, FitPlane(255));
        GL.Vertex3(FitPlane(255), 0.0f, FitPlane(255));
        GL.Vertex3(FitPlane(255), 0.0f, FitPlane(0));
        GL.Vertex3(FitPlane(0), 0.0f, FitPlane(0));
        GL.End();

        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.blue);
        GL.Vertex3(FitPlane(64), 0.0f, FitPlane(0));
        GL.Vertex3(FitPlane(64), 0.0f, FitPlane(255));
        GL.Vertex3(FitPlane(255 - 64), 0.0f, FitPlane(255));
        GL.Vertex3(FitPlane(255 - 64), 0.0f, FitPlane(0));
        GL.Vertex3(FitPlane(64), 0.0f, FitPlane(0));
        GL.End();

        DrawEnd();
    }
    public void DrawHand(Vector3[] landmarks) 
    {
        DrawBegin();

        for(int i = 0; i < landmarks.Length; ++i)
        {
            var landmark = landmarks[i];
            GL.Begin(GL.LINE_STRIP);
            GL.Color(Color.red);
            for (float theta = 0.0f; theta < (2.0f * PI); theta += 0.01f) 
            {
                DrawCircle(landmark.x, landmark.y, theta);
            }
            GL.End();
        }

        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.blue);
        GL.Vertex3(FitPlane(landmarks[0].x), 0.0f, FitPlane(landmarks[0].y));
        GL.Vertex3(FitPlane(landmarks[1].x), 0.0f, FitPlane(landmarks[1].y));
        GL.Vertex3(FitPlane(landmarks[2].x), 0.0f, FitPlane(landmarks[2].y));
        GL.Vertex3(FitPlane(landmarks[3].x), 0.0f, FitPlane(landmarks[3].y));
        GL.Vertex3(FitPlane(landmarks[4].x), 0.0f, FitPlane(landmarks[4].y));
        GL.End();

        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.blue);
        GL.Vertex3(FitPlane(landmarks[0].x), 0.0f, FitPlane(landmarks[0].y));
        GL.Vertex3(FitPlane(landmarks[5].x), 0.0f, FitPlane(landmarks[5].y));
        GL.Vertex3(FitPlane(landmarks[6].x), 0.0f, FitPlane(landmarks[6].y));
        GL.Vertex3(FitPlane(landmarks[7].x), 0.0f, FitPlane(landmarks[7].y));
        GL.Vertex3(FitPlane(landmarks[8].x), 0.0f, FitPlane(landmarks[8].y));
        GL.End();

        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.blue);
        GL.Vertex3(FitPlane(landmarks[5].x), 0.0f, FitPlane(landmarks[5].y));
        GL.Vertex3(FitPlane(landmarks[9].x), 0.0f, FitPlane(landmarks[9].y));
        GL.Vertex3(FitPlane(landmarks[10].x), 0.0f, FitPlane(landmarks[10].y));
        GL.Vertex3(FitPlane(landmarks[11].x), 0.0f, FitPlane(landmarks[11].y));
        GL.Vertex3(FitPlane(landmarks[12].x), 0.0f, FitPlane(landmarks[12].y));
        GL.End();

        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.blue);
        GL.Vertex3(FitPlane(landmarks[9].x), 0.0f, FitPlane(landmarks[9].y));
        GL.Vertex3(FitPlane(landmarks[13].x), 0.0f, FitPlane(landmarks[13].y));
        GL.Vertex3(FitPlane(landmarks[14].x), 0.0f, FitPlane(landmarks[14].y));
        GL.Vertex3(FitPlane(landmarks[15].x), 0.0f, FitPlane(landmarks[15].y));
        GL.Vertex3(FitPlane(landmarks[16].x), 0.0f, FitPlane(landmarks[16].y));
        GL.End();

        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.blue);
        GL.Vertex3(FitPlane(landmarks[13].x), 0.0f, FitPlane(landmarks[13].y));
        GL.Vertex3(FitPlane(landmarks[17].x), 0.0f, FitPlane(landmarks[17].y));
        GL.End();

        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.blue);
        GL.Vertex3(FitPlane(landmarks[0].x), 0.0f, FitPlane(landmarks[0].y));
        GL.Vertex3(FitPlane(landmarks[17].x), 0.0f, FitPlane(landmarks[17].y));
        GL.Vertex3(FitPlane(landmarks[18].x), 0.0f, FitPlane(landmarks[18].y));
        GL.Vertex3(FitPlane(landmarks[19].x), 0.0f, FitPlane(landmarks[19].y));
        GL.Vertex3(FitPlane(landmarks[20].x), 0.0f, FitPlane(landmarks[20].y));
        GL.End();

        DrawEnd();
    }

}
