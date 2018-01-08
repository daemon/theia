package net.rocketeer.mathai.utils;

import org.bytedeco.javacpp.opencv_core;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.CV_INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;

public class ImageUtils {
  public static opencv_core.IplImage thumbnail(String photoPath, int tHeight) {
    opencv_core.IplImage image = cvLoadImage(photoPath);
    int tWidth = (int) (image.width() * ((float) tHeight) / image.height());
    opencv_core.IplImage dest = cvCreateImage(cvSize(tWidth, tHeight), image.depth(), image.nChannels());
    cvResize(image, dest, CV_INTER_LINEAR);
    cvSaveImage(photoPath, dest);
    return dest;
  }
}
