package net.rocketeer.mathai.utils;

import android.content.Context;

public class DisplayUtils {
  public static float dpFromPx(Context context, float px) {
    return px / context.getResources().getDisplayMetrics().density;
  }

  public static float pxFromDp(Context context, float dp) {
    return dp * context.getResources().getDisplayMetrics().density;
  }
}
