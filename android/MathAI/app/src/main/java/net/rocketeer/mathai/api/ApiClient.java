package net.rocketeer.mathai.api;

import android.content.Context;

import com.google.gson.FieldNamingPolicy;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import net.rocketeer.mathai.R;
import net.rocketeer.mathai.api.net.AssignmentCreateResponse;
import net.rocketeer.mathai.api.net.AssignmentResponse;
import net.rocketeer.mathai.api.net.AssignmentSubmitRequest;
import net.rocketeer.mathai.api.data.AuthData;
import net.rocketeer.mathai.api.net.AuthResponse;
import net.rocketeer.mathai.api.net.CreateProfileRequest;
import net.rocketeer.mathai.api.net.CreateProfileResponse;
import net.rocketeer.mathai.api.net.CreateUserResponse;
import net.rocketeer.mathai.api.net.GradeResponse;
import net.rocketeer.mathai.api.net.ListProfileResponse;
import net.rocketeer.mathai.api.net.SimpleSuccessResponse;
import net.rocketeer.mathai.api.net.SyncAssignmentRequest;
import net.rocketeer.mathai.api.net.SyncAssignmentResponse;

import java.io.File;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava.RxJavaCallAdapterFactory;
import retrofit2.converter.gson.GsonConverterFactory;
import rx.Single;
import rx.android.schedulers.AndroidSchedulers;
import rx.schedulers.Schedulers;

public class ApiClient {
  private final Context mContext;
  private final RestApi mService;

  public ApiClient(Context context) {
    mContext = context;
    final Gson gson = new GsonBuilder().setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES).create();
    final Retrofit retrofit = new Retrofit.Builder().baseUrl(mContext.getString(R.string.base_url))
        .addConverterFactory(GsonConverterFactory.create(gson))
        .addCallAdapterFactory(RxJavaCallAdapterFactory.create())
        .build();
    mService = retrofit.create(RestApi.class);
  }

  public Single<GradeResponse> fetchGradedImage(String imagePath, int tag) {
    File file = new File(imagePath);
    RequestBody reqBody = RequestBody.create(MediaType.parse("applications/image"), file);
    MultipartBody.Part body = MultipartBody.Part.createFormData("image", file.getName(), reqBody);
    RequestBody jsonTag = RequestBody.create(MediaType.parse("applications/json"), String.valueOf(tag));
    return prepareCall(mService.gradeImage(body, jsonTag));
  }

  public Single<AuthResponse> doLogin(String email, String password) {
    return prepareCall(mService.loginUser(email, password));
  }

  public Single<CreateUserResponse> doRegister(String email, String password) {
    return prepareCall(mService.createUser(new AuthData(email, password)));
  }

  private static <T> Single<T> prepareCall(Single<T> observable) {
    return observable.observeOn(AndroidSchedulers.mainThread()).subscribeOn(Schedulers.io());
  }

  public Single<AssignmentCreateResponse> submitAssignment(AssignmentSubmitRequest request) {
    return prepareCall(mService.submitAssignment(request));
  }

  public Single<SimpleSuccessResponse> checkLogin(String token) {
    return prepareCall(mService.checkToken(token));
  }

  public Single<AssignmentResponse> fetchAssignments(String token) {
    return prepareCall(mService.fetchAssignments(token));
  }

  public Single<ListProfileResponse> listProfiles(String token) {
    return prepareCall(mService.listProfiles(token));
  }

  public Single<CreateProfileResponse> createProfile(CreateProfileRequest request) {
    return prepareCall(mService.createProfile(request));
  }

  public Single<SyncAssignmentResponse> syncAssignments(SyncAssignmentRequest request) {
    return prepareCall(mService.syncAssignments(request));
  }
}
