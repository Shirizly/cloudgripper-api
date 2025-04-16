from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m-seg.pt")  # load YOLO

results = model("recordings/dataset_0_sequence_20250203_155921/image_base_0000.png")  # run inference
# results.show()  # display results
print(results)