package net.rocketeer.mathai.widget;

import android.app.Activity;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import net.rocketeer.mathai.DetailsActivity;
import net.rocketeer.mathai.R;
import net.rocketeer.mathai.io.assignment.AssignmentReader;
import net.rocketeer.mathai.utils.Dates;

import java.text.ParseException;
import java.util.List;

public class MainViewAdapter extends RecyclerView.Adapter<MainViewAdapter.ViewHolder> {
  private final List<AssignmentReader.ReadData> mItems;
  private final Activity mActivity;

  public MainViewAdapter(Activity activity, List<AssignmentReader.ReadData> objects) {
    mActivity = activity;
    mItems = objects;
  }

  @Override
  public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
    return new ViewHolder(LayoutInflater.from(mActivity).inflate(R.layout.assignment_item, null));
  }

  @Override
  public void onBindViewHolder(ViewHolder holder, int position) {
    holder.mAsstTitle.setText(String.format("%s #%s", mActivity.getString(R.string.assignment), mItems.size() - position));
    try {
      holder.mAsstDate.setText(Dates.readableDateString(Dates.dateFromString(mItems.get(position).metadata.date)));
    } catch (ParseException e) {}

    holder.mAsstGrade.setText(Math.round(mItems.get(position).metadata.grade) + "%");
    holder.itemView.setOnClickListener(v -> {
      DetailsActivity.startDetailsActivity(mActivity, mItems.get(position).metadata.pagePaths());
    });
  }

  @Override
  public int getItemCount() {
    return mItems.size();
  }

  static class ViewHolder extends RecyclerView.ViewHolder {
    private final TextView mAsstTitle;
    private final TextView mAsstDate;
    private final TextView mAsstGrade;

    ViewHolder(View view) {
      super(view);
      mAsstTitle = view.findViewById(R.id.asstItemTitle);
      mAsstDate = view.findViewById(R.id.asstItemDate);
      mAsstGrade = view.findViewById(R.id.asstItemGrade);
    }
  }
}
