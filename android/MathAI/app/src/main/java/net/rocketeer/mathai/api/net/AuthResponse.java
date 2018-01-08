package net.rocketeer.mathai.api.net;

public class AuthResponse {
  private boolean success;
  private String token;
  private String message;

  public String token() {
    return this.token;
  }

  public String message() {
    return this.message;
  }

  public boolean success() {
    return this.success;
  }
}
