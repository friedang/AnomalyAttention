# Updated values from "mean_dist_aps"

mean_dist_aps_new = {
    "car": 0.0,
    "truck": 0.41437775549799905,
    "bus": 0.5648530196796765,
    "trailer": 0.2504245786685416,
    "construction_vehicle": 0.13681889039740514,
    "pedestrian": 0.0,
    "motorcycle": 0.3476255332753885,
    "bicycle": 0.18720096741312187,
    "traffic_cone": 0.507223168109222,
    "barrier": 0.47669931241063357
  }

# Exclude "car" and "pedestrian"
excluded_keys=["car", "pedestrian"]
filtered_values_new = [value for key, value in mean_dist_aps_new.items() if key not in excluded_keys]



# Calculate the mean of the remaining values

mean_filtered_new = sum(filtered_values_new) / len(filtered_values_new)

print(mean_filtered_new)