package net.rocketeer.mathai;

import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;

import net.rocketeer.mathai.api.GradePoint;
import net.rocketeer.mathai.api.GradeResponse;
import net.rocketeer.mathai.api.GradingClient;

import rx.android.schedulers.AndroidSchedulers;
import rx.schedulers.Schedulers;

public class GradeActivity extends AppCompatActivity {
  private ProgressDialog mDialog;
  private String mImagePath;
  private static final String MARK_CORRECT = "✓";
  private static final String MARK_INCORRECT = "✗";
  private static final Paint PAINT_CORRECT = new Paint();
  private static final Paint PAINT_INCORRECT = new Paint();
  private ImageView mImageView;

  private void initDialog() {
    mDialog = new ProgressDialog(this);
    mDialog.setMessage(getString(R.string.currently_grading));
    mDialog.setCancelable(true);
    mDialog.setOnCancelListener(e -> finish());
    mDialog.show();
  }

  public void onCreate(Bundle savedInstance) {
    super.onCreate(savedInstance);
    setContentView(R.layout.activity_grade);
    Intent intent = getIntent();
    mImagePath = intent.getStringExtra(MainActivity.GRADE_IMAGE_PATH);
    initDialog();
    mImageView = findViewById(R.id.grade_image);

    GradingClient client = new GradingClient(this);
    client.fetchGradedAssignment(mImagePath)
        .subscribeOn(Schedulers.io())
        .observeOn(AndroidSchedulers.mainThread())
        .subscribe(this::onResponse, e -> {
          Log.d("math", e.getMessage());
        });
  }

  private void onResponse(GradeResponse response) {
    mDialog.dismiss();
    Bitmap imBmp = BitmapFactory.decodeFile(mImagePath);
    Bitmap bmp = imBmp.copy(Bitmap.Config.ARGB_8888, true);
    Canvas canvas = new Canvas(bmp);
    PAINT_CORRECT.setTextSize(bmp.getWidth() / 20);
    PAINT_INCORRECT.setTextSize(bmp.getWidth() / 20);
    for (GradePoint point : response.points()) {
      Paint paint = point.isCorrect() ? PAINT_CORRECT : PAINT_INCORRECT;
      String markTxt = point.isCorrect() ? MARK_CORRECT : MARK_INCORRECT;
      canvas.drawText(markTxt, (int) (point.x() * bmp.getWidth()),
          (int) (point.y() * bmp.getHeight()), paint);
    }

    canvas.save();
    mImageView.setImageDrawable(new BitmapDrawable(getResources(), bmp));
  }

  static {
    PAINT_CORRECT.setColor(Color.GREEN);
    PAINT_INCORRECT.setColor(Color.RED);
    PAINT_CORRECT.setStyle(Paint.Style.FILL);
    PAINT_INCORRECT.setStyle(Paint.Style.FILL);
  }
}
