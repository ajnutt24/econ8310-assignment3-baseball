import os
import xml.etree.ElementTree as ET

import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


### Dataset ###

# 1. Parse the XML annotation file for a single video
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

valid_labels = {"sports ball", "baseball"}

frame_boxes = {}

for track in root.findall("track"):
    label = track.get("label", "").lower()
    if label not in valid_labels:
        continue

for box in track.findall("box"):
    if box.get("outside") == "1":
        continue

            frame_idx = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            if frame_idx not in frame_boxes:
                frame_boxes[frame_idx] = (xtl, ytl, xbr, ybr)

    return frame_boxes


# 2. Define the custom Dataset class
class BaseballDataset(Dataset):

    def __init__(self, video_dir, xml_dir, img_size=224):
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

# 3. Match video files to their XML annotation files by filename
        self.samples = []

        video_files = {
            os.path.splitext(f)[0]: os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith(".mov")
        }

        xml_files = {
            os.path.splitext(f)[0]: os.path.join(xml_dir, f)
            for f in os.listdir(xml_dir)
            if f.lower().endswith(".xml")
        }

        paired = set(video_files.keys()) & set(xml_files.keys())
        print(f"Found {len(paired)} video/annotation pairs out of {len(video_files)} videos")

# 4. Build a flat list
for name in sorted(paired):
    video_path = video_files[name]
    xml_path = xml_files[name]

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_boxes = parse_xml(xml_path)

for frame_idx, box in frame_boxes.items():
    xtl, ytl, xbr, ybr = box
    norm_box = torch.tensor([
        xtl / orig_w,
        ytl / orig_h,
        xbr / orig_w,
        ybr / orig_h,
        ], dtype=torch.float32)

        self.samples.append((video_path, frame_idx, norm_box))

        print(f"Total annotated frames: {len(self.samples)}")

# 5. Returns how many samples we have
def __len__(self):
    return len(self.samples)

# 6. Loads and returns one sample by index
def __getitem__(self, idx):
    video_path, frame_idx, box = self.samples[idx]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

if not ret:
    raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = self.transform(frame)

    return frame_tensor, box


# 7. Wrap the dataset in a DataLoader for batched training
def get_dataloader(video_dir, xml_dir, batch_size=4, shuffle=True):
    dataset = BaseballDataset(video_dir, xml_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader


### Model ###

# 1. Define the CNN architecture
class BallDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # 2. Convolutional layers 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 3. Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    # 4. Forward pass
def forward(self, x):
    x = self.conv_layers(x)
    x = self.fc_layers(x)
    return x


### Training ###

# 1. Set paths to your data folders -- update these to match your machine
video_dir = "C:/Users/slima/Downloads/RawVideos"
xml_dir = "C:/Users/slima/Downloads/Annotations"

# 2. Set training hyperparameters
epochs = 3
batch_size = 4
learning_rate = 0.001
save_path = "weights.pt"

# 3. Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 4. Load data
loader = get_dataloader(video_dir, xml_dir, batch_size=batch_size, shuffle=True)

# 5. Initialize the model and move it to the device
model = BallDetector().to(device)

# 6. Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# 7. Training loop
print(f"\nStarting training for {epochs} epochs...\n")

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for frames, boxes in loader:
        frames = frames.to(device)
        boxes = boxes.to(device)

        preds = model(frames)
        loss = loss_fn(preds, boxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch}/{epochs}]  Loss: {avg_loss:.4f}")

# 8. Save model weights to disk
torch.save(model.state_dict(), save_path)
print(f"\nWeights saved to: {save_path}")



### Import script ###

# 1. Set paths -- use one you annotated
weights_path = "weights.pt"
video_path = "C:/Users/slima/Downloads/RawVideos/IMG_8027_joel.mov"

# 2. Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Load the model and saved weights
model = BallDetector().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print(f"Model loaded from: {weights_path}")

# 4. Same transform used during training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 5. Open the video and run the model frame by frame
cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nRunning inference on: {video_path}")
print(f"{'Frame':>6}  {'xtl':>6}  {'ytl':>6}  {'xbr':>6}  {'ybr':>6}  (pixels)\n")

frame_idx = 0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        # 6. Get prediction
        pred = model(tensor).squeeze(0).cpu().tolist()

        xtl = int(pred[0] * orig_w)
        ytl = int(pred[1] * orig_h)
        xbr = int(pred[2] * orig_w)
        ybr = int(pred[3] * orig_h)

        print(f"{frame_idx:>6}  {xtl:>6}  {ytl:>6}  {xbr:>6}  {ybr:>6}")
        frame_idx += 1

cap.release()
print(f"\nDone. Processed {frame_idx} frames.")
