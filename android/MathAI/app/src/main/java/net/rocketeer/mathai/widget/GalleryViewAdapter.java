package net.rocketeer.mathai.widget;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.support.annotation.NonNull;
import android.util.DisplayMetrics;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.AnimationUtils;
import android.widget.ArrayAdapter;
import android.widget.GridView;
import android.widget.ImageView;

import net.rocketeer.mathai.DisplayDimensions;
import net.rocketeer.mathai.R;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import rx.subjects.PublishSubject;

public class GalleryViewAdapter extends ArrayAdapter<GalleryItem> {
  private final List<GalleryItem> mItems;
  private final Context mContext;
  private PublishSubject<GalleryItem> mSubject = PublishSubject.create();
  private Map<String, Bitmap> mCache = new HashMap<>();

  public GalleryViewAdapter(@NonNull Context context, int resource, @NonNull List<GalleryItem> objects) {
    super(context, resource, objects);
    mItems = objects;
    mContext = context;
  }

  public PublishSubject<GalleryItem> subject() {
    return mSubject;
  }

  public List<GalleryItem> items() {
    return mItems;
  }

  @Override
  public View getView(int position, View convertView, ViewGroup parent) {
    ImageView item;
    GalleryItem gItem = mItems.get(position);
    DisplayMetrics metrics = new DisplayDimensions((Activity) mContext).metrics();
    int offset = (int) (6 * metrics.density);
    if (gItem.isDummy()) {
      item = (ImageView) LayoutInflater.from(mContext).inflate(R.layout.gallery_add_button, null);
      item.setOnClickListener(v -> {
        mSubject.onNext(gItem);
        v.startAnimation(AnimationUtils.loadAnimation(mContext, R.anim.click_anim));
      });
    } else {
      item = new ImageView(mContext);
      item.setOnClickListener(v -> {
        new ImageDialog(mContext, gItem.photoPath()).show();
      });
      if (!mCache.containsKey(gItem.photoPath()))
        mCache.put(gItem.photoPath(), ThumbnailUtils.extractThumbnail(BitmapFactory.decodeFile(gItem.photoPath()),
            metrics.widthPixels / 2 - offset, metrics.heightPixels / 4));
      Bitmap tImage = mCache.get(gItem.photoPath());
      item.setImageBitmap(tImage);
    }
    item.setLayoutParams(new GridView.LayoutParams(metrics.widthPixels / 2 - offset, metrics.heightPixels / 4));
    return item;
  }
}
