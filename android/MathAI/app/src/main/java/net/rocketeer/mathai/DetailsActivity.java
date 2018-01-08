package net.rocketeer.mathai;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.widget.GridView;

import net.rocketeer.mathai.widget.GalleryItem;
import net.rocketeer.mathai.widget.GalleryViewAdapter;

import java.util.ArrayList;
import java.util.List;

public class DetailsActivity extends AppCompatActivity {
  private GridView mDetailsGridView;
  private GalleryViewAdapter mAdapter;

  @Override
  protected void onCreate(Bundle savedInstance) {
    super.onCreate(savedInstance);
    setContentView(R.layout.activity_details);
    String[] photoPaths = getIntent().getStringArrayExtra("photopaths");
    List<GalleryItem> gItems = new ArrayList<>();
    for (String path : photoPaths)
      gItems.add(GalleryItem.makeImage(path));
    mDetailsGridView = findViewById(R.id.detailsView);
    mAdapter = new GalleryViewAdapter(this, R.id.detailsView, gItems);
    mDetailsGridView.setAdapter(mAdapter);
  }

  public static void startDetailsActivity(Activity parent, List<String> photoPaths) {
    String[] paths = photoPaths.toArray(new String[photoPaths.size()]);
    Intent intent = new Intent(parent, DetailsActivity.class);
    intent.putExtra("photopaths", paths);
    parent.startActivity(intent);
  }
}
