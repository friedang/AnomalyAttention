# import matplotlib.pyplot as plt
# # Daten für die letzten AP-Werte pro Klasse
# ap_values_last = [0.77776, 0.40686, 0.55114, 0.19430, 0.10124, 0.76500, 0.38754, 
#                   0.20544, 0.49153, 0.46540]
# classes = ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", 
#            "motorcycle", "bicycle", "traffic_cone", "barrier"]

# # Balkendiagramm für die letzten AP-Werte erstellen
# plt.figure(figsize=(12, 6))
# plt.barh(classes, ap_values_last, color='lightcoral')
# plt.xlabel('AP')
# plt.title('Average Precision (AP) per Class')
# plt.gca().invert_yaxis()
# plt.show()

import matplotlib.pyplot as plt

# Daten für die Klassenverteilung
dist = {'car': 19053, 'truck': 3926, 'bus': 680, 'trailer': 1575, 'vehicle.construction': 835, 'pedestrian': 9075, 'motorcycle': 509, 'bicycle': 634, 'trafficcone': 4462, 'barrier': 6598}

labels = list(dist.keys())
sizes = list(dist.values())
# labels = ['Car', 'Pedestrian', 'Barrier', 'Traffic Cone', 'Truck', 'Trailer', 'Motorcycle', 'Construction', 'Bus', 'Bicycle']
# sizes = [490000, 210000, 152000, 98000, 88000, 24000, 12000, 14000, 15000, 12000]

# Tortendiagramm erstellen
plt.figure(figsize=(10, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution')
plt.savefig("/workspace/CenterPoint/class_dist_5_1.png")
