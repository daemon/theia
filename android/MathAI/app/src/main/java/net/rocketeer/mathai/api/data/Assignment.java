package net.rocketeer.mathai.api.data;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Assignment {
  private int id;
  private int authorId;
  private double grade;
  private String date;

  public int id() {
    return this.id;
  }

  public int authorId() {
    return this.authorId;
  }

  public double grade() {
    return this.grade;
  }

  public Date date() {
    DateFormat df = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
    try {
      return df.parse(this.date);
    } catch (ParseException e) {
      return null;
    }
  }
}
