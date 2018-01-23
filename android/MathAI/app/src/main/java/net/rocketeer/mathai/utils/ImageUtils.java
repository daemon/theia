package net.rocketeer.mathai.utils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ImageWriter;

import java.io.FileOutputStream;

public class ImageUtils {
  public static Bitmap thumbnail(String photoPath, int tHeight) {
    BitmapFactory.Options options = new BitmapFactory.Options();
    options.inJustDecodeBounds = false;
    options.inPreferredConfig = Bitmap.Config.RGB_565;
    options.inDither = true;
    Bitmap bMap = BitmapFactory.decodeFile(photoPath, options);
    int tWidth = (int) (bMap.getWidth() * ((float) tHeight) / bMap.getHeight());
    bMap = Bitmap.createScaledBitmap(bMap, tWidth, tHeight, true);

    FileOutputStream stream = null;
    try {
      stream = new FileOutputStream(photoPath);
      bMap.compress(Bitmap.CompressFormat.JPEG, 90, stream);
    } catch (Exception ignored) {
      return null;
    }
    return bMap;
  }
}
