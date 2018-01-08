package net.rocketeer.mathai.api.net;

import net.rocketeer.mathai.api.data.Assignment;

import java.util.List;

public class AssignmentResponse {
  private List<Assignment> assignments;

  public List<Assignment> assignments() {
    return this.assignments;
  }
}
