package net.rocketeer.mathai.io.assignment;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.gson.Gson;

import net.rocketeer.mathai.api.data.GradePoint;
import net.rocketeer.mathai.api.net.AssignmentCreateResponse;
import net.rocketeer.mathai.api.net.GradeResponse;
import net.rocketeer.mathai.io.LocalAuthStore;
import net.rocketeer.mathai.utils.Dates;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.List;

public class AssignmentWriter {
  private final List<GradeResponse> mResponses;
  private final List<String> mImagePaths;

  private static final String MARK_CORRECT = "✓";
  private static final String MARK_INCORRECT = "✗";
  private static final Paint PAINT_CORRECT = new Paint();
  private static final Paint PAINT_INCORRECT = new Paint();
  static final String ASST_PATH = "assignments";
  private final Context mContext;
  private final int mAssignId;
  private final LocalAuthStore mAuthStore;
  private final List<Integer> mWorksheetIds;

  public AssignmentWriter(Context context, List<String> imagePaths, List<GradeResponse> responses,
                          AssignmentCreateResponse createResponse) {
    mImagePaths = imagePaths;
    mResponses = responses;
    mContext = context;
    mAssignId = createResponse.id;
    mWorksheetIds = createResponse.worksheetIds;
    mAuthStore = new LocalAuthStore(context);
  }

  public static void drawGrades(AssignmentMetadata assignment, List<GradeResponse> responses) {
    for (int i = 0; i < responses.size(); ++i) {
      Worksheet wkst = assignment.worksheets.get(i);
      drawGrade(wkst.pagePath + ".copy", wkst.pagePath, responses.get(i));
    }
  }

  public static Bitmap drawGrade(String imagePath, String imagePathOut, GradeResponse response) {
    Bitmap imBmp = BitmapFactory.decodeFile(imagePath);
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
    try (FileOutputStream fOut = new FileOutputStream(new File(imagePathOut))) {
      bmp.compress(Bitmap.CompressFormat.JPEG, 80, fOut);
    } catch (IOException ignored) {}
    return bmp;
  }

  public AssignmentMetadata write(int profileId) throws IOException {
    String ts = Dates.currentDate();
    final File file = new File(mContext.getFilesDir() + File.separator + ASST_PATH +
        File.separator + profileId + File.separator + ts);
    file.mkdirs();
    double gradeSum = 0;
    double gradeLength = 0;
    List<Worksheet> worksheets = new LinkedList<>();
    for (int i = 0; i < mResponses.size(); ++i) {
      String path = mImagePaths.get(i);
      GradeResponse response = mResponses.get(i);

      File f = new File(path);
      String pathOut = file.getAbsolutePath() + File.separator + f.getName();
      drawGrade(path, pathOut, response);
      gradeSum += response.grade() * response.points().size();
      gradeLength += response.points().size();

      File newFile = new File(file.getAbsolutePath() + File.separator + f.getName() + ".copy");
      FileUtils.copyFile(f, newFile);

      int wkstId = mWorksheetIds.get(i);
      worksheets.add(new Worksheet(wkstId, pathOut));
      f.delete();
    }

    File mdFile = new File(file.getAbsolutePath() + File.separator + "base.metadatas");
    double grade = gradeLength == 0 ? 0 : gradeSum / gradeLength;
    int pId = mAuthStore.currentProfileId();
    AssignmentMetadata metadata = new AssignmentMetadata(mAssignId, grade, Dates.currentDate(), pId, worksheets);
    writeMetadata(metadata, mdFile);
    return metadata;
  }

  public static void writeMetadata(AssignmentMetadata metadata, File metadataFile) throws FileNotFoundException {
    try (PrintWriter writer = new PrintWriter(metadataFile)) {
      Gson gson = new Gson();
      writer.print(gson.toJson(metadata));
    }
  }

  static {
    PAINT_CORRECT.setColor(Color.GREEN);
    PAINT_INCORRECT.setColor(Color.RED);
    PAINT_CORRECT.setStyle(Paint.Style.FILL);
    PAINT_INCORRECT.setStyle(Paint.Style.FILL);
  }
}
