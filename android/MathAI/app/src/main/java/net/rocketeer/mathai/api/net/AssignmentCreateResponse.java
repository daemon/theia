package net.rocketeer.mathai.api.net;

import java.util.List;

public class AssignmentCreateResponse {
  public final int id;
  public final List<Integer> worksheetIds;

  public AssignmentCreateResponse(int id, List<Integer> worksheetIds) {
    this.id = id;
    this.worksheetIds = worksheetIds;
  }

  public int id() {
    return this.id;
  }
}
