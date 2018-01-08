package net.rocketeer.mathai;

import android.app.Activity;
import android.content.Context;
import android.util.DisplayMetrics;

public class DisplayDimensions {
  private final DisplayMetrics mMetrics;

  public DisplayDimensions(Context context) {
    DisplayMetrics metrics = context.getResources().getDisplayMetrics();
    mMetrics = metrics;
  }

  public DisplayMetrics metrics() {
    return mMetrics;
  }

  public double inchesToPixels(double inches) {
    return mMetrics.densityDpi * inches;
  }
}
