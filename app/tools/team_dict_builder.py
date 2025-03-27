import os
import pandas as pd
import csv
from itertools import chain


def compile_team_names_from_files(data_directory: str) -> set[str]:
    file_list = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))
                if "HomeTeam" in df.columns:
                    df = df.rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"})
                file_list.append(df[["Home", "Away"]])

    teams_df = pd.concat(file_list)

    return set(teams_df["Home"]).union(teams_df["Away"])


def find_partial_match(team, source_list):
    team_parts = team.split()
    for other_team in source_list:
        if other_team.startswith(team_parts[0]):
            return other_team
    return None


team_dict = pd.read_csv("team_dict.csv")
fbref_team_names = compile_team_names_from_files("DATA/FBREF")
fbduk_team_names = compile_team_names_from_files("DATA/FBDUK")
oddsportal_team_names = pd.read_csv("fixtures.csv")[["home", "away"]]
oddsportal_team_names = set(oddsportal_team_names["home"]).union(
    oddsportal_team_names["away"]
)

# Exact matches
all_teams = fbref_team_names | fbduk_team_names | oddsportal_team_names

dict_teams = list(chain(*team_dict.values.tolist()))
unmatched_teams = {team: ("", "", "") for team in all_teams if team not in dict_teams}

exact_matches = {
    team: (team, team, team)
    for team in unmatched_teams
    if team in fbref_team_names
    and team in fbduk_team_names
    and team in oddsportal_team_names
}

team_mappings = {}

for team in unmatched_teams.keys():
    if team in exact_matches:
        continue

    fbref_match = (
        team if team in fbref_team_names else find_partial_match(team, fbref_team_names)
    )
    fbduk_match = (
        team if team in fbduk_team_names else find_partial_match(team, fbduk_team_names)
    )
    oddsportal_match = (
        team
        if team in oddsportal_team_names
        else find_partial_match(team, oddsportal_team_names)
    )

    if fbref_match and fbduk_match and oddsportal_match:
        team_mappings[team] = (fbref_match, fbduk_match, oddsportal_match)
    else:
        unmatched_teams[team] = (fbref_match, fbduk_match, oddsportal_match)

# Print unmatched teams for manual checking
if unmatched_teams:
    print("Teams needing manual review:")
    for team, (fbref, fbduk, oddsportal) in unmatched_teams.items():
        print(
            f"Standard: {team} | FBREF: {fbref} | FBDUK: {fbduk} | OddsPortal: {oddsportal}"
        )

fieldnames = ["standard_name", "fbref", "fbduk", "oddsportal"]

with open("unmatched_team_dict.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for std_name, (fbref, fbduk, oddsportal) in unmatched_teams.items():
        writer.writerow(
            {
                "standard_name": std_name,
                "fbref": fbref,
                "fbduk": fbduk,
                "oddsportal": oddsportal,
            }
        )

with open("team_dict.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for std_name, (fbref, fbduk, oddsportal) in team_mappings.items():
        writer.writerow(
            {
                "standard_name": std_name,
                "fbref": fbref,
                "fbduk": fbduk,
                "oddsportal": oddsportal,
            }
        )

    for std_name, (fbref, fbduk, oddsportal) in exact_matches.items():
        writer.writerow(
            {
                "standard_name": std_name,
                "fbref": fbref,
                "fbduk": fbduk,
                "oddsportal": oddsportal,
            }
        )

pd.read_csv("unmatched_team_dict.csv").sort_values("standard_name").to_csv(
    "unmatched_team_dict.csv", index=False
)

pd.read_csv("team_dict.csv").drop_duplicates(
    subset=["standard_name", "fbref", "fbduk", "oddsportal"], keep="first"
).sort_values("standard_name").to_csv("team_dict.csv", index=False)
