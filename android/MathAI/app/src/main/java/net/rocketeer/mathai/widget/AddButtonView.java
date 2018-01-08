package net.rocketeer.mathai.widget;

import android.app.Activity;
import android.content.Context;
import android.graphics.PixelFormat;
import android.opengl.GLSurfaceView;

import net.rocketeer.mathai.DisplayDimensions;
import net.rocketeer.mathai.widget.AddButton;

public class AddButtonView extends GLSurfaceView {
  private final AddButton mRenderer;

  public AddButtonView(Activity context) {
    super(context);
    setEGLContextClientVersion(2);
    mRenderer = new AddButton((float) new DisplayDimensions(context).inchesToPixels(0.05));
    setZOrderOnTop(true);
    setEGLConfigChooser(8, 8, 8, 8, 16, 0);
    getHolder().setFormat(PixelFormat.TRANSLUCENT);
    setRenderer(mRenderer);
    setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
  }
}
