package net.rocketeer.mathai;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.TabLayout;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RelativeLayout;

import com.tomergoldst.tooltips.ToolTip;
import com.tomergoldst.tooltips.ToolTipsManager;

import net.rocketeer.mathai.api.ApiClient;
import net.rocketeer.mathai.api.data.Profile;
import net.rocketeer.mathai.io.LocalAuthStore;
import net.rocketeer.mathai.widget.AlertOKDialog;
import net.rocketeer.mathai.widget.OKDialogEnum;

import java.net.ConnectException;
import java.net.SocketTimeoutException;
import java.util.Collections;

import static net.rocketeer.mathai.LoginActivity.FormState.LOGIN;
import static net.rocketeer.mathai.LoginActivity.FormState.REGISTER;

public class LoginActivity extends AppCompatActivity {
  private EditText mPasswordEdit;
  private EditText mPasswordEditConfirm;
  private EditText mEmailEdit;
  private Button mSubmitButton;
  private TabLayout mAuthTabLayout;
  private ToolTipsManager mTtManager;
  private RelativeLayout mRootLayout;
  private ToolTip mTtPass1;
  private ToolTip mTtPass2;

  private FormState mState = REGISTER;
  private int mLastPasswordLength = 0;
  private boolean mIsConfShown = false;
  private ApiClient mApiClient;
  private ProgressDialog mDialog;
  private LocalAuthStore mStore;

  public enum FormState { REGISTER, LOGIN }

  private void showPasswordToolTip() {
    if (mPasswordEdit.length() >= 8)
      return;
    mTtManager.show(mTtPass1);
  }

  private String passwordString() {
    return mPasswordEdit.getText().toString();
  }

  private String passwordConfirmString() {
    return mPasswordEditConfirm.getText().toString();
  }

  private void showConfirmToolTip() {
    if (mPasswordEdit.length() == 0 || passwordString().equals(passwordConfirmString())) {
      mTtManager.findAndDismiss(mPasswordEditConfirm);
      mIsConfShown = false;
    } else if (!mIsConfShown) {
      mTtManager.show(mTtPass2);
      mIsConfShown = true;
    }
  }

  private void setupHandlers() {
    mPasswordEdit.setOnFocusChangeListener((view, onFocus) -> {
      if (onFocus && mState == REGISTER)
        showPasswordToolTip();
      else
        mTtManager.findAndDismiss(mPasswordEdit);
    });
    mPasswordEdit.addTextChangedListener(new TextWatcher() {
      public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {}

      @Override
      public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {
        if (isFormValid())
          mSubmitButton.setEnabled(true);
        int passLen = mPasswordEdit.length();
        if (passLen >= 8)
          mTtManager.findAndDismiss(mPasswordEdit);
        else if (passLen < 8 && mLastPasswordLength == 8)
          showPasswordToolTip();
        mLastPasswordLength = passLen;
      }
      public void afterTextChanged(Editable editable) {}
    });

    mPasswordEditConfirm.addTextChangedListener(new TextWatcher() {
      public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {}
      public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {
        if (isFormValid())
          mSubmitButton.setEnabled(true);
        showConfirmToolTip();
      }
      public void afterTextChanged(Editable editable) {}
    });

    mAuthTabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
      @Override
      public void onTabSelected(TabLayout.Tab tab) {
        if (tab.getPosition() == 0)
          inflateRegister();
        else
          inflateLogin();
      }
      public void onTabUnselected(TabLayout.Tab tab) {}
      public void onTabReselected(TabLayout.Tab tab) {}
    });

    mSubmitButton.setOnClickListener(e -> {
      mDialog.show();
      if (mState == LOGIN)
        mApiClient.doLogin(emailString(), passwordString())
            .subscribe(response -> {
              mDialog.hide();
              if (!response.success())
                new AlertOKDialog(this, response.message(), getString(R.string.error)).show();
              else
                startMainActivity(response.token(), emailString(), passwordString());
            }, error -> {
              if (error instanceof ConnectException || error instanceof SocketTimeoutException)
                OKDialogEnum.NO_CONNECTION.show(this);
              mDialog.hide();
            });
      else
        mApiClient.doRegister(emailString(), passwordString())
            .subscribe(response -> {
              mDialog.hide();
              if (!response.success())
                new AlertOKDialog(this, response.message(), getString(R.string.error)).show();
              else {
                mStore.currentProfileId(response.profileId());
                mStore.profiles(Collections.singletonList(new Profile(response.profileId(), "Default")));
                startMainActivity(response.token(), emailString(), passwordString());
              }
            }, error -> {
              if (error instanceof ConnectException || error instanceof SocketTimeoutException)
                OKDialogEnum.NO_CONNECTION.show(this);
              mDialog.hide();
            });
    });
  }

  private void startMainActivity(String token, String username, String password) {
    Intent intent = new Intent(this, MainActivity.class);
    intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
    mStore.token(token);
    mStore.password(password);
    mStore.email(username);
    startActivity(intent);
    finish();
  }

  private String emailString() {
    return mEmailEdit.getText().toString();
  }

  private boolean isFormValid() {
    if (mState == REGISTER)
      return passwordString().equals(passwordConfirmString()) && passwordString().length() >= 8 &&
          mEmailEdit.getText().length() > 0;
    else
      return passwordString().length() >= 8 && mEmailEdit.getText().length() > 0;
  }

  private void inflateRegister() {
    if (mState == REGISTER)
      return;
    mState = REGISTER;
    mPasswordEditConfirm.setText("");
    mPasswordEditConfirm.setVisibility(View.VISIBLE);
  }

  private void inflateLogin() {
    if (mState == LOGIN)
      return;
    mState = LOGIN;
    mTtManager.findAndDismiss(mPasswordEdit);
    mTtManager.findAndDismiss(mPasswordEditConfirm);
    mIsConfShown = false;
    mPasswordEditConfirm.setVisibility(View.INVISIBLE);
  }

  private void tryLogin() {
    String email = mStore.email();
    String password = mStore.password();
    String token = mStore.token();
    if (email == null || password == null || token == null) {
      resumeOnCreate();
      return;
    }

    mApiClient.checkLogin(token).subscribe(response -> {
      if (response.success())
        startMainActivity(token, email, password);
      else
        mApiClient.doLogin(email, password).subscribe(r2 -> {
          if (response.success())
            startMainActivity(r2.token(), email, password);
          else
            resumeOnCreate();
        }, ignored -> startMainActivity(token, email, password));
    }, ignored -> startMainActivity(token, email, password));
  }

  private void resumeOnCreate() {
    setContentView(R.layout.activity_auth);
    mPasswordEdit = findViewById(R.id.editPassword);
    mPasswordEditConfirm = findViewById(R.id.editPasswordConfirm);
    mEmailEdit = findViewById(R.id.editEmail);
    mRootLayout = findViewById(R.id.loginForm);
    mAuthTabLayout = findViewById(R.id.authTabLayout);
    mSubmitButton = findViewById(R.id.submitButton);

    mTtManager = new ToolTipsManager();
    mTtPass1 = new ToolTip.Builder(this, mPasswordEdit, mRootLayout,
        getString(R.string.pass_req_tip), ToolTip.POSITION_ABOVE)
        .setBackgroundColor(getResources().getColor(R.color.colorAccent)).build();
    mTtPass2 = new ToolTip.Builder(this, mPasswordEditConfirm, mRootLayout,
        getString(R.string.pass_conf_tip), ToolTip.POSITION_ABOVE)
        .setBackgroundColor(getResources().getColor(R.color.colorAccent)).build();

    mDialog = new ProgressDialog(this);
    mDialog.setMessage(getString(R.string.submitting));
    mDialog.setCancelable(false);

    setupHandlers();
  }

  @Override
  public void onCreate(Bundle savedInstance) {
    super.onCreate(savedInstance);
    ActivityCompat.requestPermissions(this,
        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
    ActivityCompat.requestPermissions(this,
        new String[]{Manifest.permission.CAMERA}, 0);

    mApiClient = new ApiClient(this);
    mStore = new LocalAuthStore(this);
    tryLogin();
  }
}
