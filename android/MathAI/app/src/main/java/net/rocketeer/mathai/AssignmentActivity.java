package net.rocketeer.mathai;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.GridView;

import net.rocketeer.mathai.api.ApiClient;
import net.rocketeer.mathai.api.net.AssignmentSubmitRequest;
import net.rocketeer.mathai.api.net.GradeResponse;
import net.rocketeer.mathai.io.LocalAuthStore;
import net.rocketeer.mathai.io.assignment.AssignmentMetadata;
import net.rocketeer.mathai.io.assignment.AssignmentWriter;
import net.rocketeer.mathai.widget.GalleryItem;
import net.rocketeer.mathai.widget.GalleryViewAdapter;
import net.rocketeer.mathai.io.SingleShotImageAction;
import net.rocketeer.mathai.widget.OKDialogEnum;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class AssignmentActivity extends AppCompatActivity {
  private SingleShotImageAction mAction;
  private List<GalleryItem> mGalleryItems = new LinkedList<>();
  private GalleryViewAdapter mAdapter;
  private GridView mGalleryView;
  private ApiClient mClient;
  private LocalAuthStore mAuthStore;

  @Override
  protected void onActivityResult(int reqCode, int resCode, Intent bundle) {
    if (resCode != RESULT_OK)
      return;
    mAdapter.items().add(mAdapter.items().size() - 1, GalleryItem.makeImage(mAction.photoFile().getAbsolutePath()));
    mGalleryView.setAdapter(mAdapter);
    mGalleryView.invalidateViews();
  }

  private void onGrade(View view) {
    ProgressDialog dialog = new ProgressDialog(this);
    dialog.setCancelable(false);
    dialog.setMessage(getString(R.string.currently_grading));
    dialog.show();
    List<GradeResponse> responses = new ArrayList<>();
    List<String> paths = new ArrayList<>();
    List<String> gradeTokens = new LinkedList<>();
    int profileId = mAuthStore.currentProfileId();
    for (GalleryItem item : mAdapter.items())
      if (!item.isDummy())
        paths.add(item.photoPath());
    int i = 0;
    for (GalleryItem item : mAdapter.items()) {
      if (item.isDummy())
        continue;
      ++i;
      mClient.fetchGradedImage(item.photoPath(), i).subscribe(response -> {
        responses.add(response);
        if (responses.size() != paths.size())
          return;
        Collections.sort(responses, ((r1, r2) -> r1.tag() - r2.tag()));
        for (GradeResponse r : responses)
          gradeTokens.add(r.token());
        mClient.submitAssignment(new AssignmentSubmitRequest(gradeTokens, mAuthStore.token(), mAuthStore.currentProfileId())).subscribe(response2 -> {
          AssignmentWriter writer = new AssignmentWriter(this, paths, responses, response2);
          AssignmentMetadata metadata;
          try {
            metadata = writer.write(profileId);
          } catch (IOException e) {
            finish();
            return;
          }
          dialog.dismiss();
          if (metadata != null)
            DetailsActivity.startDetailsActivity(this, metadata.pagePaths());
          finish();
        }, e1 -> {
          OKDialogEnum.NO_CONNECTION.show(this);
          dialog.dismiss();
        });
      }, e -> {
        OKDialogEnum.NO_CONNECTION.show(this);
        dialog.dismiss();
      });
    }
  }

  @Override
  protected void onCreate(Bundle bundle) {
    super.onCreate(bundle);
    setContentView(R.layout.activity_assignment);
    Bundle extras = getIntent().getExtras();
    if (extras != null) {
      String photoPath = extras.getString("assignment");
      if (photoPath != null)
        mGalleryItems.add(GalleryItem.makeImage(photoPath));
    }

    mClient = new ApiClient(this);
    mGalleryItems.add(GalleryItem.makeDummy());
    mGalleryView = findViewById(R.id.galleryView);
    mAdapter = new GalleryViewAdapter(this, R.layout.activity_assignment, mGalleryItems);
    mAdapter.subject().subscribe(gItem -> {
      if (!gItem.isDummy())
        return;
      mAction = new SingleShotImageAction(this, false);
      mAction.execute();
    });
    mGalleryView.setAdapter(mAdapter);

    mAuthStore = new LocalAuthStore(this);
    FloatingActionButton gradeButton = findViewById(R.id.gradeBtn);
    gradeButton.setOnClickListener(this::onGrade);
  }
}
