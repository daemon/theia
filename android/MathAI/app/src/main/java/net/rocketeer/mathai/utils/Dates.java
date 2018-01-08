package net.rocketeer.mathai.utils;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Dates {
  public static final DateFormat DATE_FORMAT = new SimpleDateFormat("yyyyMMdd_HHmmss");

  public static Date dateFromString(String string) throws ParseException {
    DateFormat format = DATE_FORMAT;
    return format.parse(string);
  }

  public static String stringFromDate(Date date) {
    DateFormat format = DATE_FORMAT;
    return format.format(date);
  }

  public static String currentDate() {
    return DATE_FORMAT.format(new Date());
  }

  public static String readableDateString(Date date) {
    DateFormat format = new SimpleDateFormat("yyyy/MM/dd");
    return format.format(date);
  }
}
