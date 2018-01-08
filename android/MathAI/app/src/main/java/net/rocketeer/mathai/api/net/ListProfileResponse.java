package net.rocketeer.mathai.api.net;

import net.rocketeer.mathai.api.data.Profile;

import java.util.List;

public class ListProfileResponse {
  public final List<Profile> profiles;
  ListProfileResponse(List<Profile> profiles) {
    this.profiles = profiles;
  }
}
