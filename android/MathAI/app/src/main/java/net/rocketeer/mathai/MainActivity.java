package net.rocketeer.mathai;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.DividerItemDecoration;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.Toolbar;
import android.view.View;

import net.rocketeer.mathai.api.ApiClient;
import net.rocketeer.mathai.io.assignment.AssignmentReader;
import net.rocketeer.mathai.io.SingleShotImageAction;
import net.rocketeer.mathai.io.LocalAuthStore;
import net.rocketeer.mathai.utils.ImageUtils;
import net.rocketeer.mathai.widget.MainViewAdapter;
import net.rocketeer.mathai.widget.ProfileDrawerBuilder;

import java.io.FileNotFoundException;
import java.util.List;

public class MainActivity extends AppCompatActivity {
  public static final String GRADE_IMAGE_PATH = "mathai.grade.image";
  private String mToken;
  private ApiClient mClient;
  private LocalAuthStore mStore;
  private SingleShotImageAction mAction;
  private RecyclerView mRecyclerView;
  private MainViewAdapter mAdapter;

  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    mClient = new ApiClient(this);
    mStore = new LocalAuthStore(this);
    String token = mStore.token();
    String username = mStore.email();
    String password = mStore.password();
    mToken = token;
    if (token == null && (username == null || password == null))
      startLoginActivity();
    else if (token == null)
      doLogin(username, password);
    mClient.checkLogin(token).subscribe(response -> {
      if (!response.success() && (username == null || password == null))
        startLoginActivity();
      else
        doLogin(username, password);
    });

    mClient.listProfiles(mToken).subscribe(response -> {
      mStore.profiles(response.profiles);
      ProfileDrawerBuilder builder = new ProfileDrawerBuilder(username).withProfiles(response.profiles);
      builder.withOnSwitchProfileListener(profile -> {
        mStore.currentProfileId(profile.id);
        populateView();
      });
      builder.withActivity(this).build();
    });

    Toolbar toolbar = findViewById(R.id.mainToolbar);
    toolbar.setTitle(R.string.assignments);
    toolbar.setBackgroundColor(getResources().getColor(R.color.colorPrimary));
    toolbar.setTitleTextColor(Color.WHITE);
    setSupportActionBar(toolbar);
    initAddButton();
    populateView();
  }

  @Override
  protected void onResume() {
    super.onResume();
    populateView();
  }

  private void populateView() {
    mRecyclerView = findViewById(R.id.asstList);
    List<AssignmentReader.ReadData> assignments;
    int profileId = mStore.currentProfileId();
    try {
      assignments = AssignmentReader.readAllAssignments(this, profileId);
    } catch (FileNotFoundException e) {
      return;
    }

    View noneIndicator = findViewById(R.id.main_none_text);
    mAdapter = new MainViewAdapter(this, assignments);
    if (assignments.size() == 0) {
      mRecyclerView.setAdapter(mAdapter);
      noneIndicator.setVisibility(View.VISIBLE);
      return;
    }
    noneIndicator.setVisibility(View.GONE);
    mRecyclerView.setAdapter(mAdapter);

    LinearLayoutManager layoutManager = new LinearLayoutManager(this);
    mRecyclerView.setLayoutManager(layoutManager);
    DividerItemDecoration deco = new DividerItemDecoration(mRecyclerView.getContext(),
        layoutManager.getOrientation());
    deco.setDrawable(getResources().getDrawable(R.drawable.list_border));
    mRecyclerView.addItemDecoration(deco);
  }

  private void doLogin(String username, String password) {
    mClient.doLogin(username, password).subscribe(response -> {
      if (response.success())
        mToken = response.token();
      else
        startLoginActivity();
    });
  }

  private void startLoginActivity() {
    Intent intent = new Intent(this, LoginActivity.class);
    intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
    startActivity(intent);
    finish();
  }

  @Override
  protected void onActivityResult(int reqCode, int resCode, Intent data) {
    if (resCode != RESULT_OK)
      return;
    ImageUtils.thumbnail(mAction.photoFile().getAbsolutePath(), 800);
    Intent intent = new Intent(this, AssignmentActivity.class);
    intent.putExtra("assignment", mAction.photoFile().getAbsolutePath());
    startActivity(intent);
  }

  private void initAddButton() {
    FloatingActionButton button = findViewById(R.id.add_btn);
    button.setOnClickListener(v -> {
      mAction = new SingleShotImageAction(this, true);
      mAction.execute();
    });
  }
}
