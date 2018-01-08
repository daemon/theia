package net.rocketeer.mathai.io.assignment;

import android.content.Context;

import com.google.gson.Gson;

import net.rocketeer.mathai.utils.Dates;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.text.ParseException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class AssignmentReader {
  private final String mDirectory;

  public AssignmentReader(String directory) {
    mDirectory = directory;
  }

  public AssignmentMetadata read() throws FileNotFoundException {
    FileReader reader = new FileReader(mDirectory + File.separator + "base.metadatas");
    Gson gson = new Gson();
    return gson.fromJson(reader, AssignmentMetadata.class);
  }

  public static List<ReadData> readAssignments(Context context, int profileId, int limit, int pageNo) {
    List<ReadData> metadatas = new LinkedList<>();
    File[] files = new File(context.getFilesDir() + File.separator + AssignmentWriter.ASST_PATH + File.separator + profileId).listFiles();
    if (files == null)
      return metadatas;
    Arrays.sort(files, (f1, f2) -> {
      try {
        return (int) (0.001 * Dates.dateFromString(f2.getName()).getTime() - 0.001 * Dates.dateFromString(f1.getName()).getTime());
      } catch (ParseException ignored) {}
      return 0;
    });
    int counter = 0;
    for (File file : files) {
      if (counter != pageNo * limit) {
        ++counter;
        continue;
      }
      if (metadatas.size() == limit)
        break;
      AssignmentMetadata metadata = null;
      try {
        metadata = new AssignmentReader(file.getAbsolutePath()).read();
      } catch (FileNotFoundException ignored) {}
      if (metadata != null)
        metadatas.add(new ReadData(metadata, file));
    }
    return metadatas;
  }

  public static List<ReadData> readAllAssignments(Context context, int profileId) throws FileNotFoundException {
    return readAssignments(context, profileId, Integer.MAX_VALUE, 0);
  }

  public static class ReadData {
    public final AssignmentMetadata metadata;
    public final File file;
    public ReadData(AssignmentMetadata metadata, File file) {
      this.metadata = metadata;
      this.file = file;
    }
  }
}
