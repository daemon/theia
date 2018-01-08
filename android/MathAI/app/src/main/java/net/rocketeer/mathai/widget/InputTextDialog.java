package net.rocketeer.mathai.widget;

import android.app.ActionBar;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.support.constraint.ConstraintLayout;
import android.support.v7.widget.LinearLayoutCompat;
import android.view.ViewGroup;
import android.widget.EditText;

import net.rocketeer.mathai.R;
import net.rocketeer.mathai.utils.DisplayUtils;

public class InputTextDialog {
  private final EditText mEditText;
  private final Context mContext;
  private final AlertDialog.Builder mAlert;

  public InputTextDialog(Context context, String title, String hint) {
    mAlert = new AlertDialog.Builder(context);
    LinearLayoutCompat layout = new LinearLayoutCompat(context);
    layout.setLayoutParams(new LinearLayoutCompat.LayoutParams(
        LinearLayoutCompat.LayoutParams.MATCH_PARENT, LinearLayoutCompat.LayoutParams.MATCH_PARENT));

    int padPx = (int) DisplayUtils.pxFromDp(context, 15);
    layout.setPadding(padPx, padPx, padPx, 0);
    mEditText = new EditText(context);
    mEditText.setHint(hint);
    mAlert.setTitle(title);
    layout.addView(mEditText, new LinearLayoutCompat.LayoutParams(LinearLayoutCompat.LayoutParams.MATCH_PARENT,
        LinearLayoutCompat.LayoutParams.WRAP_CONTENT));
    mAlert.setView(layout);
    mContext = context;
  }

  public void setSubmitListener(SubmitListener listener) {
    mAlert.setPositiveButton(mContext.getString(R.string.submit), new DialogInterface.OnClickListener() {
      public void onClick(DialogInterface dialog, int whichButton) {
        listener.onSubmit(mEditText.getText().toString());
      }
    });
  }

  public void show() {
    mAlert.show();
  }

  @FunctionalInterface
  public interface SubmitListener {
    void onSubmit(String input);
  }
}
