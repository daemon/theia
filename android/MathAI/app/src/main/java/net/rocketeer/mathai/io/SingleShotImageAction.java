package net.rocketeer.mathai.io;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;

public class SingleShotImageAction {
  private final Activity mActivity;
  private final boolean mInit;
  private File mPhotoFile;

  public SingleShotImageAction(Activity activity, boolean init) {
    mActivity = activity;
    mInit = init;
  }

  public File photoFile() {
    return this.mPhotoFile;
  }

  public void execute() {
    Intent pictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
    if (pictureIntent.resolveActivity(mActivity.getPackageManager()) != null) {
      try {
        mPhotoFile = createImageFile();
      } catch (IOException e) {
        return;
      }
      Uri photoURI = FileProvider.getUriForFile(mActivity, "net.rocketeer.mathai.fileprovider", mPhotoFile);
      pictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
      mActivity.startActivityForResult(pictureIntent, 1);
    }
  }

  private File createImageFile() throws IOException {
    String ts = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    File storageDir = new File(mActivity.getExternalFilesDir(Environment.DIRECTORY_PICTURES) + File.separator + "tmp");
    storageDir.mkdirs();
    if (mInit)
      for (File file : storageDir.listFiles())
        file.delete();
    return File.createTempFile(ts, ".jpg", storageDir);
  }
}
