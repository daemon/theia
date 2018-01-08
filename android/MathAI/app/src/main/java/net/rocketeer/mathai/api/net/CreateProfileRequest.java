package net.rocketeer.mathai.api.net;

public class CreateProfileRequest {
  private final String name;
  private final String token;

  public CreateProfileRequest(String token, String name) {
    this.token = token;
    this.name = name;
  }
}
