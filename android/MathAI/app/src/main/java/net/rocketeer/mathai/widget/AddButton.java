package net.rocketeer.mathai.widget;

import android.app.Activity;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;

import net.rocketeer.mathai.DisplayDimensions;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class AddButton implements GLSurfaceView.Renderer {
  private final String vertexShaderCode =
      "uniform mat4 uMVP;" +
          "attribute vec4 vPosition;" +
          "void main() {" +
          "  gl_Position = uMVP * vPosition;" +
          "}";

  private final String fragmentShaderCode =
      "precision mediump float;" +
          "uniform vec4 vColor;" +
          "void main() {" +
          "  gl_FragColor = vColor;" +
          "}";
  static float boxCoords[] = {
      -0.6f, -0.95f, 0,
      -0.6f, 0.95f, 0,
      0.6f, 0.95f, 0,
      0.6f, -0.95f, 0,
      -0.6f, -0.95f, 0,
  };
  private final float[] mMVPMatrix = new float[16];
  private final float[] mProjectionMatrix = new float[16];
  private final float[] mViewMatrix = new float[16];
  private final float mLineWidth;

  private int mProgram;
  private FloatBuffer mBoxBuffer = fromBuffer(boxCoords);

  public AddButton(float lineWidth) {
    mLineWidth = lineWidth;
  }

  private static FloatBuffer fromBuffer(float[] buffer) {
    ByteBuffer buf = ByteBuffer.allocateDirect(4 * buffer.length);
    buf.order(ByteOrder.nativeOrder());
    FloatBuffer fb = buf.asFloatBuffer();
    fb.put(buffer);
    fb.position(0);
    return fb;
  }

  public static int loadShader(int type, String shaderCode) {
    int shader = GLES20.glCreateShader(type);
    GLES20.glShaderSource(shader, shaderCode);
    GLES20.glCompileShader(shader);
    return shader;
  }

  @Override
  public void onSurfaceCreated(GL10 gl10, EGLConfig eglConfig) {
    GLES20.glDisable(GLES20.GL_DITHER);
    GLES20.glClearColor(0, 0, 0, 0);
    int vtxShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode);
    int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode);
    mProgram = GLES20.glCreateProgram();

    GLES20.glAttachShader(mProgram, vtxShader);
    GLES20.glAttachShader(mProgram, fragmentShader);
    GLES20.glLinkProgram(mProgram);
  }

  @Override
  public void onSurfaceChanged(GL10 gl10, int width, int height) {
    GLES20.glViewport(0, 0, width, height);
    float ratio = (float) width / height;
    Matrix.frustumM(mProjectionMatrix, 0, -ratio, ratio, -1, 1, 3, 7);
  }

  @Override
  public void onDrawFrame(GL10 gl10) {
    Matrix.setLookAtM(mViewMatrix, 0, 0, 0, -3, 0f, 0f, 0f, 0f, 1.0f, 0.0f);
    Matrix.multiplyMM(mMVPMatrix, 0, mProjectionMatrix, 0, mViewMatrix, 0);
    GLES20.glUseProgram(mProgram);
    int hndPos = GLES20.glGetAttribLocation(mProgram, "vPosition");
    GLES20.glEnableVertexAttribArray(hndPos);
    GLES20.glVertexAttribPointer(hndPos, 3, GLES20.GL_FLOAT, false, 12, mBoxBuffer);
    int hndColor = GLES20.glGetUniformLocation(mProgram, "vColor");
    GLES20.glUniform4fv(hndColor, 1, new float[]{0, 0, 0, 0.2f}, 0);
    int mMVPMatrixHandle = GLES20.glGetUniformLocation(mProgram, "uMVP");
    GLES20.glUniformMatrix4fv(mMVPMatrixHandle, 1, false, mMVPMatrix, 0);
    GLES20.glLineWidth(mLineWidth);
    GLES20.glDrawArrays(GLES20.GL_LINE_STRIP, 0, boxCoords.length / 3);
    GLES20.glDisableVertexAttribArray(hndPos);
  }
}
