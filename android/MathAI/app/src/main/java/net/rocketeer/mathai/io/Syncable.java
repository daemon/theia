package net.rocketeer.mathai.io;

public abstract class Syncable {
  private FinishSyncListener mSyncListener;
  private FinishSyncListener mListener;

  public abstract void sync(FinishSyncListener listener);

  public void setOnFinishSyncListener(FinishSyncListener listener) {
    mListener = listener;
  }

  public void runSync() {
    sync(mListener);
  }
}
