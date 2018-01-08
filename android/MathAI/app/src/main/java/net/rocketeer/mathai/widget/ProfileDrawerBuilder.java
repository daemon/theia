package net.rocketeer.mathai.widget;

import android.app.Activity;
import android.app.ProgressDialog;

import com.mikepenz.materialdrawer.AccountHeader;
import com.mikepenz.materialdrawer.AccountHeaderBuilder;
import com.mikepenz.materialdrawer.Drawer;
import com.mikepenz.materialdrawer.DrawerBuilder;
import com.mikepenz.materialdrawer.model.PrimaryDrawerItem;
import com.mikepenz.materialdrawer.model.ProfileDrawerItem;
import com.mikepenz.materialdrawer.model.SectionDrawerItem;

import net.rocketeer.mathai.R;
import net.rocketeer.mathai.api.ApiClient;
import net.rocketeer.mathai.api.data.Profile;
import net.rocketeer.mathai.api.net.CreateProfileRequest;
import net.rocketeer.mathai.io.LocalAuthStore;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ProfileDrawerBuilder extends DrawerBuilder {
  private final String mEmail;
  private List<Profile> mProfiles = new ArrayList<>();
  private ProfileSwitchListener mListener;

  public ProfileDrawerBuilder(String email) {
    mEmail = email;
  }

  public ProfileDrawerBuilder withProfiles(List<Profile> profiles) {
    mProfiles.addAll(profiles);
    return this;
  }

  public ProfileDrawerBuilder withOnSwitchProfileListener(ProfileSwitchListener listener) {
    mListener = listener;
    return this;
  }

  private PrimaryDrawerItem itemFromProfile(Profile profile, LocalAuthStore store) {
    PrimaryDrawerItem item = new PrimaryDrawerItem().withName(profile.name);
    item.withSetSelected(store.currentProfileId() == profile.id);
    item.withOnDrawerItemClickListener((v, pos, it) -> {
      if (mListener != null)
        mListener.accept(profile);
      return true;
    });
    return item;
  }

  public Drawer build() {
    final Activity activity = mActivity;
    ApiClient client = new ApiClient(activity);
    LocalAuthStore store = new LocalAuthStore(activity);
    AccountHeader header = new AccountHeaderBuilder()
        .withActivity(activity)
        .withHeaderBackground(R.drawable.profile_bkgd)
        .addProfiles(new ProfileDrawerItem().withEmail(mEmail)).build();
    withAccountHeader(header);
    withFullscreen(false);

    PrimaryDrawerItem addItem = new PrimaryDrawerItem()
        .withName("Add profile");
    addDrawerItems(addItem);
    addDrawerItems(new SectionDrawerItem().withName(R.string.profiles));

    for (Profile profile : mProfiles)
      addDrawerItems(itemFromProfile(profile, store));

    Drawer drawer = super.build();
    addItem.withOnDrawerItemClickListener((view, pos, dItem) -> {
      InputTextDialog dialog = new InputTextDialog(activity, activity.getString(R.string.new_profile),
          activity.getString(R.string.profile_name));
      ProgressDialog pDialog = new ProgressDialog(activity);
      pDialog.setMessage(activity.getString(R.string.submitting));
      pDialog.show();
      dialog.setSubmitListener(name -> {
            client.createProfile(new CreateProfileRequest(store.token(), name)).subscribe(response -> {
              List<Profile> profiles = store.profiles();
              Profile profile = new Profile(response.profileId, name);
              profiles.add(profile);
              store.profiles(profiles);
              drawer.addItem(itemFromProfile(profile, store));
              pDialog.dismiss();
            });
          });
      dialog.show();
      return true;
    });
    return drawer;
  }

  @FunctionalInterface
  public interface ProfileSwitchListener {
    void accept(Profile profile);
  }
}
