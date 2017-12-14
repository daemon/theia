package net.rocketeer.mathai;

import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends AppCompatActivity {
  public static final String GRADE_IMAGE_PATH = "mathai.grade.image";
  private String mPhotoPath;

  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Toolbar toolbar = findViewById(R.id.main_toolbar);
    toolbar.setTitle(R.string.assignments);
    toolbar.setBackgroundColor(getResources().getColor(R.color.colorPrimary));
    toolbar.setTitleTextColor(Color.WHITE);
    setSupportActionBar(toolbar);
    initAddButton();
  }

  private File createImageFile() throws IOException {
    String ts = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
    File image = File.createTempFile(ts, ".jpg", storageDir);

    mPhotoPath = image.getAbsolutePath();
    return image;
  }

  @Override
  protected void onActivityResult(int reqCode, int resCode, Intent data) {
    if (resCode != RESULT_OK)
      return;
    Intent gradeIntent = new Intent(this, GradeActivity.class);
    gradeIntent.putExtra(GRADE_IMAGE_PATH, mPhotoPath);
    startActivity(gradeIntent);
  }

  private void initAddButton() {
    FloatingActionButton button = findViewById(R.id.add_btn);
    button.setOnClickListener(v -> {
      Intent pictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
      if (pictureIntent.resolveActivity(getPackageManager()) != null) {
        File photoFile = null;
        try {
          photoFile = createImageFile();
        } catch (IOException e) {}
        if (photoFile == null)
          return;
        Uri photoURI = FileProvider.getUriForFile(this, "net.rocketeer.mathai.fileprovider", photoFile);
        pictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
        startActivityForResult(pictureIntent, 1);
      }
    });
  }
}
