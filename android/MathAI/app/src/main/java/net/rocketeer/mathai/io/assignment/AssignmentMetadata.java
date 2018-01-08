package net.rocketeer.mathai.io.assignment;

import java.util.LinkedList;
import java.util.List;

public class AssignmentMetadata {
  public final double grade;
  public final String date;
  public final int id;
  public final int profileId;
  public final List<Worksheet> worksheets;

  public AssignmentMetadata(int id, double grade, String date, int profileId, List<Worksheet> worksheets) {
    this.grade = grade;
    this.date = date;
    this.profileId = profileId;
    this.worksheets = worksheets;
    this.id = id;
  }

  public List<String> pagePaths() {
    List<String> pagePaths = new LinkedList<>();
    for (Worksheet wkst : worksheets)
      pagePaths.add(wkst.pagePath);
    return pagePaths;
  }
}
