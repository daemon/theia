package net.rocketeer.mathai.widget;

import android.content.Context;

import net.rocketeer.mathai.R;

public enum OKDialogEnum {
  NO_CONNECTION(R.string.no_connection, R.string.error),
  INVALID_PASSWORD(R.string.wrong_password, R.string.error);
  private final int mTitleResId;
  private final int mMessageResId;

  OKDialogEnum(int messageResId, int titleResId) {
    mMessageResId = messageResId;
    mTitleResId = titleResId;
  }

  public void show(Context context) {
    AlertOKDialog dialog = new AlertOKDialog(context, context.getString(mMessageResId), context.getString(mTitleResId));
    dialog.show();
  }
}
