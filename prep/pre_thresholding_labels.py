import pandas as pd

data = pd.read_csv("../data/test_data_4_10_15.csv", delimiter=",")

print "# Thresholding labels ..."

d_a = []
d_f = []
d_o = []

for idx, movie in data.iterrows():
    a_labels = []
    f_labels = []
    o_labels = []

    if movie.get("occasion_boys_night", 0.0) >= 3.0:
        o_labels.append("occasion_boys_night")

    if movie.get("occasion_date", 0.0) >= 3.0:
        o_labels.append("occasion_date")

    if movie.get("occasion_distraction", 0.0) >= 3.0:
        o_labels.append("occasion_distraction")

    if movie.get("occasion_family", 0.0) >= 3.0:
        o_labels.append("occasion_family")

    if movie.get("occasion_friends", 0.0) >= 3.0:
        o_labels.append("occasion_friends")

    if movie.get("occasion_girls_night", 0.0) >= 3.0:
        o_labels.append("occasion_girls_night")

    if movie.get("occasion_relax", 0.0) >= 3.0:
        o_labels.append("occasion_relax")

    if movie.get("occasion_time_burning", 0.0) >= 3.0:
        o_labels.append("occasion_time_burning")

    if movie.get("atmosphere_action", 0.0) >= 3.0:
        a_labels.append("atmosphere_action")

    if movie.get("atmosphere_brutal", 0.0) >= 3.0:
        a_labels.append("atmosphere_brutal")

    if movie.get("atmosphere_dark", 0.0) >= 3.0:
        a_labels.append("atmosphere_dark")

    if movie.get("atmosphere_emotional", 0.0) >= 3.0:
        a_labels.append("atmosphere_emotional")

    if movie.get("atmosphere_food_for_thought", 0.0) >= 3.0:
        a_labels.append("atmosphere_food_for_thought")

    if movie.get("atmosphere_funny", 0.0) >= 3.0:
        a_labels.append("atmosphere_funny")

    if movie.get("atmosphere_romantic", 0.0) >= 3.0:
        a_labels.append("atmosphere_romantic")

    if movie.get("atmosphere_thrilling", 0.0) >= 3.0:
        a_labels.append("atmosphere_thrilling")

    if movie.get("feeling_agitated", 0.0) >= 3.0:
        f_labels.append("feeling_agitated")

    if movie.get("feeling_euphoric", 0.0) >= 3.0:
        f_labels.append("feeling_euphoric")

    if movie.get("feeling_happy", 0.0) >= 3.0:
        f_labels.append("feeling_happy")

    if movie.get("feeling_listless", 0.0) >= 3.0:
        f_labels.append("feeling_listless")

    if movie.get("feeling_sad", 0.0) >= 3.0:
        f_labels.append("feeling_sad")

    if movie.get("feeling_strained", 0.0) >= 3.0:
        f_labels.append("feeling_strained")

    if movie.get("feeling_stressed", 0.0) >= 3.0:
        f_labels.append("feeling_stressed")

    if movie.get("feeling_unsatisfied", 0.0) >= 3.0:
        f_labels.append("feeling_unsatisfied")

    if len(o_labels) > 0:
        d_o.append({
                    'feelyt_id': movie["feelyt_id"],
                    'descr': movie["description_en"],
                    'labels': ','.join(o_labels)
                })

    if len(a_labels) > 0:
        d_a.append({
                    'feelyt_id': movie["feelyt_id"],
                    'descr': movie["description_en"],
                    'labels': ','.join(a_labels)
                })

    if len(f_labels) > 0:
        d_f.append({
                    'feelyt_id': movie["feelyt_id"],
                    'descr': movie["description_en"],
                    'labels': ','.join(f_labels)
                })

df = pd.DataFrame(d_a)
df.to_csv('../data/atmosphere_train.csv', index=False)
