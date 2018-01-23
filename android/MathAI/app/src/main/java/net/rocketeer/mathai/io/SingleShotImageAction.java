package net.rocketeer.mathai.io;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class SingleShotImageAction {
  private final Activity mActivity;
  private final boolean mInit;
  private File mPhotoFile;

  public SingleShotImageAction(Activity activity, boolean init) {
    this(activity, init, new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date()));
  }

  public SingleShotImageAction(Activity activity, boolean init, String photoPath) {
    mActivity = activity;
    mInit = init;
    try {
      mPhotoFile = createImageFile(photoPath);
    } catch (IOException e) {}
  }

  public File photoFile() {
    return this.mPhotoFile;
  }

  public void execute() {
    Intent pictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
    if (pictureIntent.resolveActivity(mActivity.getPackageManager()) != null) {
      Uri photoURI = Uri.fromFile(mPhotoFile);
      pictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
      mActivity.startActivityForResult(pictureIntent, 1);
    }
  }

  private File createImageFile(String photoPath) throws IOException {
    File storageDir = new File(mActivity.getExternalFilesDir(Environment.DIRECTORY_PICTURES) + File.separator + "tmp");
    storageDir.mkdirs();
    if (mInit)
      for (File file : storageDir.listFiles())
        file.delete();
    return File.createTempFile(photoPath, ".jpg", storageDir);
  }
}
