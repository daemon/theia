package net.rocketeer.mathai.io.assignment;

import android.content.Context;

import net.rocketeer.mathai.api.ApiClient;
import net.rocketeer.mathai.api.net.SyncAssignmentRequest;
import net.rocketeer.mathai.io.FinishSyncListener;
import net.rocketeer.mathai.io.LocalAuthStore;
import net.rocketeer.mathai.io.Syncable;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.List;

public class AssignmentSynchronizer extends Syncable {
  private final Context mContext;
  private final List<AssignmentReader.ReadData> mReadDataList;
  private final ApiClient mClient;
  private final String mToken;

  public AssignmentSynchronizer(Context context, List<AssignmentReader.ReadData> readDataList) {
    mContext = context;
    mReadDataList = readDataList;
    mClient = new ApiClient(context);
    mToken = new LocalAuthStore(context).token();
  }

  @Override
  public void sync(FinishSyncListener listener) {
    List<AssignmentMetadata> assignments = new LinkedList<>();
    for (AssignmentReader.ReadData readData : mReadDataList)
      assignments.add(readData.metadata);
    SyncAssignmentRequest request = new SyncAssignmentRequest(mToken, assignments);
    mClient.syncAssignments(request).subscribe(response -> {
      for (int i = 0; i < response.metadatas.size(); ++i)
        try {
          AssignmentWriter.drawGrades(response.metadatas.get(i), response.gradeResponses);
          AssignmentWriter.writeMetadata(response.metadatas.get(i),
              new File(mReadDataList.get(i).file.getAbsolutePath()));
        } catch (FileNotFoundException e) {
          e.printStackTrace();
        }
      listener.onFinish();
    });
  }
}
