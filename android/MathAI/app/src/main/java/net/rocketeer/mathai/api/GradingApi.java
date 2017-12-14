package net.rocketeer.mathai.api;

import okhttp3.MultipartBody;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import rx.Single;

public interface GradingApi {
  @Multipart
  @POST("grade")
  Single<GradeResponse> gradeImage(@Part MultipartBody.Part image);
}
