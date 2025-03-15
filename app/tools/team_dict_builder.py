import os
import pandas as pd


def compile_team_names_from_files(data_directory: str) -> set[str]:
    file_list = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith(".csv"):
                file_list.append(
                    pd.read_csv(os.path.join(root, file))[["Home", "Away"]]
                )

    teams_df = pd.concat(file_list)

    return set(teams_df["Home"]).union(teams_df["Away"])


fbref_team_names = compile_team_names_from_files("DATA")

oddsportal_team_names = pd.read_csv("fixtures.csv")[["home", "away"]]
oddsportal_team_names = set(oddsportal_team_names["home"]).union(
    oddsportal_team_names["away"]
)

fbref_difference = list(fbref_team_names.difference(oddsportal_team_names))
oddsportal_difference = list(oddsportal_team_names.difference(fbref_team_names))

fbref_first_words = {team.split(" ")[0]: team for team in fbref_difference}

matched_pairs = {
    team_a: fbref_first_words.get(team_a.split(" ")[0], None)
    for team_a in oddsportal_difference
}

team_dict = pd.DataFrame(
    matched_pairs.items(), columns=["oddsportal", "fbref"]
).sort_values("fbref")

file_exists = os.path.exists("team_dict.csv")

if file_exists:
    existing_team_dict = pd.read_csv("team_dict.csv")
    team_dict = team_dict[
        ~team_dict["oddsportal"].isin(existing_team_dict["oddsportal"])
        & ~team_dict["fbref"].isin(existing_team_dict["fbref"])
    ]

    i = input(
        f"Do you want to add the following teams to the dictionary? (Y/n): \r\n {team_dict}"
    )

    if i == "Y":
        team_dict.to_csv("team_dict.csv", mode="a", header=not file_exists, index=False)
    else:
        pass

print(team_dict)
