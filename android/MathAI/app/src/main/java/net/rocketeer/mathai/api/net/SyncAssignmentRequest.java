package net.rocketeer.mathai.api.net;

import net.rocketeer.mathai.io.assignment.AssignmentMetadata;

import java.util.List;

public class SyncAssignmentRequest {
  private final String token;
  private final List<AssignmentMetadata> assignments;

  public SyncAssignmentRequest(String token, List<AssignmentMetadata> assignments) {
    this.token = token;
    this.assignments = assignments;
  }
}
