package net.rocketeer.mathai.api.net;

public class CreateProfileResponse {
  public final int profileId;
  public final int ownerId;

  CreateProfileResponse(int profileId, int ownerId) {
    this.profileId = profileId;
    this.ownerId = ownerId;
  }
}
