package net.rocketeer.mathai.widget;

import java.util.Observable;

public class GalleryItem {
  private final boolean mIsDummyItem;
  private String mPhotoPath;

  GalleryItem(boolean isDummyItem) {
    mIsDummyItem = isDummyItem;
  }

  GalleryItem(String photoPath) {
    mPhotoPath = photoPath;
    mIsDummyItem = false;
  }

  public String photoPath() {
    return mPhotoPath;
  }

  public static GalleryItem makeImage(String photoPath) {
    return new GalleryItem(photoPath);
  }

  public static GalleryItem makeDummy() {
    return new GalleryItem(true);
  }

  public boolean isDummy() {
    return mIsDummyItem;
  }
}
