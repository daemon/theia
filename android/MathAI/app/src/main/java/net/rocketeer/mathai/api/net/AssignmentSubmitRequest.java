package net.rocketeer.mathai.api.net;

import java.util.List;

public class AssignmentSubmitRequest {
  private final String authToken;
  private final List<String> gradeTokens;
  private final int profileId;
  private final long durationMs;

  public AssignmentSubmitRequest(List<String> gradeTokens, String authToken, int profileId, long durationMs) {
    this.authToken = authToken;
    this.gradeTokens = gradeTokens;
    this.profileId = profileId;
    this.durationMs = durationMs;
  }
}
