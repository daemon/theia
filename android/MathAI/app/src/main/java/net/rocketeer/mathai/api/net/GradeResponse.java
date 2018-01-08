package net.rocketeer.mathai.api.net;

import net.rocketeer.mathai.api.data.GradePoint;

import java.util.List;

public class GradeResponse {
  private List<GradePoint> points;
  private String token;
  private double grade;
  private int tag;

  public List<GradePoint> points() {
    return this.points;
  }

  public String token() {
    return this.token;
  }

  public double grade() {
    return this.grade;
  }

  public int tag() {
    return this.tag;
  }
}
