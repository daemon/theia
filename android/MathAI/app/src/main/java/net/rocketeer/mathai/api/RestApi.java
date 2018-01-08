package net.rocketeer.mathai.api;

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

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Query;
import rx.Single;

public interface RestApi {
  @Multipart
  @POST("/grade")
  Single<GradeResponse> gradeImage(@Part MultipartBody.Part image, @Part("tag") RequestBody tag);

  @POST("/user/")
  Single<CreateUserResponse> createUser(@Body AuthData data);

  @GET("/user")
  Single<AuthResponse> loginUser(@Query("email") String email, @Query("password") String password);

  @POST("/assignment/")
  Single<AssignmentCreateResponse> submitAssignment(@Body AssignmentSubmitRequest request);

  @GET("/assignment/")
  Single<AssignmentResponse> fetchAssignments(@Query("token") String token);

  @GET("/sync_assignments/")
  Single<SyncAssignmentResponse> syncAssignments(@Body SyncAssignmentRequest request);

  @GET("/check_token/")
  Single<SimpleSuccessResponse> checkToken(@Query("token") String token);

  @GET("/profile/")
  Single<ListProfileResponse> listProfiles(@Query("token") String token);

  @POST("/profile/")
  Single<CreateProfileResponse> createProfile(@Body CreateProfileRequest request);
}
