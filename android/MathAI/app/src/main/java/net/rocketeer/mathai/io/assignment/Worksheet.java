package net.rocketeer.mathai.io.assignment;

public class Worksheet {
  public final String pagePath;
  public final int id;

  public Worksheet(int id, String pagePath) {
    this.pagePath = pagePath;
    this.id = id;
  }
}
