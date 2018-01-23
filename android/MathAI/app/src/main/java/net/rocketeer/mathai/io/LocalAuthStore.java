package net.rocketeer.mathai.io;

import android.content.Context;
import android.content.SharedPreferences;

import com.google.gson.Gson;

import net.rocketeer.mathai.api.ApiClient;
import net.rocketeer.mathai.api.data.Profile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static android.content.Context.MODE_PRIVATE;

public class LocalAuthStore {
  private final ApiClient mClient;
  private final Context mContext;
  private final SharedPreferences mPrefs;

  public LocalAuthStore(Context context) {
    mClient = new ApiClient(context);
    mContext = context;
    mPrefs = mContext.getSharedPreferences("mathai", MODE_PRIVATE);
  }

  public Context context() {
    return mContext;
  }

  public void token(String token) {
    mPrefs.edit().putString("token", token).apply();
  }

  public void password(String password) {
    mPrefs.edit().putString("password", password).apply();
  }

  public void email(String email) {
    mPrefs.edit().putString("email", email).apply();
  }

  public String email() {
    return mPrefs.getString("email", null);
  }

  public String password() {
    return mPrefs.getString("password", null);
  }

  public String token() {
    return mPrefs.getString("token", null);
  }

  public List<Profile> profiles() {
    Gson gson = new Gson();
    String profileStr = mPrefs.getString("profiles", null);
    if (profileStr == null)
      return null;
    return new ArrayList<>(Arrays.asList(gson.fromJson(profileStr, Profile[].class)));
  }

  public void profiles(List<Profile> profiles) {
    Gson gson = new Gson();
    Profile[] profileArr = new Profile[profiles.size()];
    profiles.toArray(profileArr);
    mPrefs.edit().putString("profiles", gson.toJson(profileArr)).apply();
  }

  public Profile currentProfile() {
    for (Profile profile : profiles())
      if (profile.id == currentProfileId())
        return profile;
    return null;
  }

  public int currentProfileId() {
    return mPrefs.getInt("profileid", -1);
  }

  public void currentProfileId(int profileId) {
    mPrefs.edit().putInt("profileid", profileId).apply();
  }
}
