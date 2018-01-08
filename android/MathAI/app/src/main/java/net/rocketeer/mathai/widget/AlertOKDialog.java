package net.rocketeer.mathai.widget;

import android.app.AlertDialog;
import android.content.Context;
import android.os.Build;

public class AlertOKDialog {
  private final AlertDialog.Builder mBuilder;

  public AlertOKDialog(Context context, String message, String title) {
    AlertDialog.Builder builder;
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
      builder = new AlertDialog.Builder(context, android.R.style.Theme_Material_Dialog_Alert);
    } else {
      builder = new AlertDialog.Builder(context);
    }
    builder.setMessage(message).setTitle(title).setNeutralButton(android.R.string.ok, (dialog, which) -> {});
    mBuilder = builder;
  }

  public void show() {
    mBuilder.show();
  }
}
