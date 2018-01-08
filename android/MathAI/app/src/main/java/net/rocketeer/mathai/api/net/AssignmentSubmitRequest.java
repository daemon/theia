package net.rocketeer.mathai.api.net;

import java.util.List;

public class AssignmentSubmitRequest {
  private final String authToken;
  private final List<String> gradeTokens;
  private final int profileId;

  public AssignmentSubmitRequest(List<String> gradeTokens, String authToken, int profileId) {
    this.authToken = authToken;
    this.gradeTokens = gradeTokens;
    this.profileId = profileId;
  }
}
