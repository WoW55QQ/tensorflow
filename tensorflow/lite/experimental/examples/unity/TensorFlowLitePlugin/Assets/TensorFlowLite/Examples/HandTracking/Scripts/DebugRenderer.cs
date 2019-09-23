using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DebugRenderer : MonoBehaviour {
    private float adjScale = 0.039f;
    private float adjPos = -5.0f;

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

    private void DrawBegin() 
    {
        CreateLineMaterial();
        lineMaterial.SetPass(0);
        GL.PushMatrix();
        GL.MultMatrix(transform.localToWorldMatrix);
    }
    private void DrawEnd() { GL.PopMatrix(); }
    private float AdjPos(float pos) { return pos * adjScale + adjPos; }

    public void DrawInput(float[,,,] inputs) 
    {
        DrawBegin();
        Color color = Color.black;
        for (int y = 0; y < height; ++y) 
        {
            GL.Begin(GL.LINE_STRIP);
            for (int x = 0; x < width; ++x) 
            {
                int pos = (y * width + x) * 3;
                color.r = inputs[0, y, x, 0];
                color.g = inputs[0, y, x, 1];
                color.b = inputs[0, y, x, 2];
                GL.Color(color);
                GL.Vertex3(AdjPos(x), 0.0f, AdjPos(y));
            }
            GL.End();
        }
        DrawEnd();
    }

    public void DrawPalm(Rect box, Vector2[] keypoints) 
    {
        var xMin = box.x * width;
        var xMax = (box.x + box.width) * width;
        var yMin = box.y * height;
        var yMax = (box.y + box.height) * height;

        DrawBegin();
        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.green);
        GL.Vertex3(AdjPos(xMin), 0.0f, AdjPos(yMin));
        GL.Vertex3(AdjPos(xMin), 0.0f, AdjPos(yMax));
        GL.Vertex3(AdjPos(xMax), 0.0f, AdjPos(yMax));
        GL.Vertex3(AdjPos(xMax), 0.0f, AdjPos(yMin));
        GL.Vertex3(AdjPos(xMin), 0.0f, AdjPos(yMin));
        GL.End();
/*
        GL.Begin(GL.LINE_STRIP);
        GL.Color(Color.red);
        for(int i = 0; i < keypoints.Length; ++i)
        {
            var x = keypoints[i].x * width;
            var y = keypoints[i].y * height;
            GL.Vertex3(AdjPos(x), 0.0f, AdjPos(y));
        }
        GL.End();
*/
        DrawEnd();
    }
}
