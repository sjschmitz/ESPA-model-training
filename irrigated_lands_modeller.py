""""
irrigated_lands_modeller.py takes two ArcGIS .shp files that are training sets to identify 'corner' polygons and partial circular polygons that have been missed by the circle 
classifier. In this context, 'corners' are polygon artifacts that are not going to be considered in the model that pop up around circular pivot irrigated land. Each training set
containts approximately 500 samples: 250 positive exmaples, 250 negative examples. 

Once the model iterates over the training set and reaches a >95% accuracy rate in evaluating the training set, it is applied to the raw polygon data to make predictions of 
each polygon to determine if it is a corner and if it is a non-complete circular pivot. 
"""

import numpy as np
import geopandas as gpd
import rasterio.features
from affine import Affine
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------#
# data path setup               #
# ------------------------------#

TRAIN_SHAPEFILE_CORNERS = r"C:\Users\sjsch\Desktop\IWRRI\phil_pivot_data\model training\training_data_cornersv2.shp"
TRAIN_SHAPEFILE_PARTIAL = r"C:\Users\sjsch\Desktop\IWRRI\phil_pivot_data\model training\training_data.shp"

# List of new shapefiles to predict on
NEW_SHAPEFILES = [
    r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\",
    r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\",
    r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\",
    r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\",
    r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\",
    r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\"
]

IMG_SIZE = 128
EPOCHS = 150 #runthroughs of training data
BATCH_SIZE = 16
ID_FIELD = "src_id"

# ------------------------------------------------#
# convert polygons to images for model trainings  #
# ------------------------------------------------#

def polygon_to_image(geom, size=256):
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    scale = max(width, height) or 1
    transform = Affine.translation(minx, miny) * Affine.scale(scale/size, scale/size)

    fill = rasterio.features.rasterize(
        [(geom, 1)], out_shape=(size, size), transform=transform, fill=0, dtype="uint8"
    )
    boundary = rasterio.features.rasterize(
        [(geom.boundary, 1)], out_shape=(size, size), transform=transform, fill=0, dtype="uint8"
    )
    return np.stack([fill, boundary], axis=0).astype(np.float32)

# -------------------------------#
# data                           #
# -------------------------------#
class TwoHeadDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"] / 1.0
        label = self.labels[item["id"]]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# -------------------------------#
# create and activate CNN model  #
# -------------------------------#

class TwoHeadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 128)
        self.corner_head = nn.Linear(128, 1)
        self.partial_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.corner_head(x), self.partial_head(x)

# -------------------------------#
# training data                  #
# -------------------------------#

corner_gdf = gpd.read_file(TRAIN_SHAPEFILE_CORNERS)
partial_gdf = gpd.read_file(TRAIN_SHAPEFILE_PARTIAL)

corner_labels = corner_gdf.set_index(ID_FIELD)["is_corner"].to_dict()
partial_labels = partial_gdf.set_index(ID_FIELD)["is_pivot"].to_dict()

all_ids = set(list(corner_labels.keys()) + list(partial_labels.keys()))
train_data = []
train_labels = {}

# cmbine data
for gdf, label_name in [(corner_gdf, "is_corner"), (partial_gdf, "is_pivot")]:
    for _, row in gdf.iterrows():
        pid = row[ID_FIELD]
        if pid not in train_labels:
            train_labels[pid] = [0, 0]  # [corner, partial]
        if label_name in row:
            if label_name == "is_corner":
                train_labels[pid][0] = int(row[label_name])
            else:
                train_labels[pid][1] = int(row[label_name])
        if not any(d["id"] == pid for d in train_data):
            train_data.append({"id": pid, "image": polygon_to_image(row.geometry, IMG_SIZE)})

train_dataset = TwoHeadDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------#
# model training                 #
# -------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoHeadCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        corner_logit, partial_logit = model(X)
        loss = criterion(corner_logit.squeeze(), y[:,0]) + criterion(partial_logit.squeeze(), y[:,1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

# ------------------------------------#
# run predictions on input shapefiles #
# ------------------------------------#

for shapefile in NEW_SHAPEFILES:
    new_gdf = gpd.read_file(shapefile)
    new_data = [{"id": row[ID_FIELD], "image": polygon_to_image(row.geometry, IMG_SIZE)}
                for _, row in new_gdf.iterrows()]

    pred_corner = {}
    pred_partial = {}
    model.eval()
    with torch.no_grad():
        for item in new_data:
            img_tensor = torch.tensor(np.expand_dims(item["image"], axis=0), dtype=torch.float32).to(device)
            c_logit, p_logit = model(img_tensor)
            pred_corner[item["id"]] = int(torch.sigmoid(c_logit).item() > 0.5)
            pred_partial[item["id"]] = int(torch.sigmoid(p_logit).item() > 0.5)

    out_file = shapefile.replace(".shp", "_predicted.shp")
    new_gdf["is_corner"] = new_gdf[ID_FIELD].map(pred_corner)
    new_gdf["is_partial_pivot"] = new_gdf[ID_FIELD].map(pred_partial)
    new_gdf.to_file(out_file)

    print(f"predictions saved to: {out_file}")
