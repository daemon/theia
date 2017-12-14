package net.rocketeer.mathai.api;

import android.content.Context;
import android.widget.Toast;

import com.google.gson.FieldNamingPolicy;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import net.rocketeer.mathai.R;

import java.io.File;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava.RxJavaCallAdapterFactory;
import retrofit2.converter.gson.GsonConverterFactory;
import rx.Single;

public class GradingClient {
  private final Context mContext;
  private final GradingApi mService;

  public GradingClient(Context context) {
    mContext = context;
    final Gson gson = new GsonBuilder().setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES).create();
    final Retrofit retrofit = new Retrofit.Builder().baseUrl(mContext.getString(R.string.base_url))
        .addCallAdapterFactory(RxJavaCallAdapterFactory.create())
        .addConverterFactory(GsonConverterFactory.create(gson))
        .build();
    mService = retrofit.create(GradingApi.class);
  }

  public Single<GradeResponse> fetchGradedAssignment(String imagePath) {
    File file = new File(imagePath);
    RequestBody reqBody = RequestBody.create(MediaType.parse("applications/image"), file);
    MultipartBody.Part body = MultipartBody.Part.createFormData("image", file.getName(), reqBody);
    return mService.gradeImage(body);
  }
}
