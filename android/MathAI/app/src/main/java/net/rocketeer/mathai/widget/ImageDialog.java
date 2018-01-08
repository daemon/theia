package net.rocketeer.mathai.widget;

import android.app.Dialog;
import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;

import net.rocketeer.mathai.R;

public class ImageDialog {
  private final Dialog mDialog;

  public ImageDialog(Context context, String photoPath) {
    mDialog = new Dialog(context);
    mDialog.getWindow().requestFeature(Window.FEATURE_NO_TITLE);
    View contentView = LayoutInflater.from(context).inflate(R.layout.gallery_item_image, null);
    ImageView imView = contentView.findViewById(R.id.gallery_item_imageview);
    imView.setImageDrawable(Drawable.createFromPath(photoPath));
    mDialog.setContentView(contentView);

    mDialog.getWindow().setLayout(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT);
    mDialog.setOnDismissListener(l -> {});
  }

  public void show() {
    mDialog.show();
  }
}
