package net.rocketeer.mathai.api.net;

import net.rocketeer.mathai.io.assignment.AssignmentMetadata;

import java.util.List;

public class SyncAssignmentResponse {
  public final List<AssignmentMetadata> metadatas;
  public final List<GradeResponse> gradeResponses;

  public SyncAssignmentResponse(List<AssignmentMetadata> metadatas, List<GradeResponse> gradeResponses) {
    this.metadatas = metadatas;
    this.gradeResponses = gradeResponses;
  }
}
