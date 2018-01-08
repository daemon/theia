package net.rocketeer.mathai.api.net;

public class CreateUserResponse {
  private boolean success;
  private String token;
  private String message;
  private Integer profileId;

  public String token() {
    return this.token;
  }

  public String message() {
    return this.message;
  }

  public boolean success() {
    return this.success;
  }

  public Integer profileId() {
    return this.profileId;
  }
}
