from detect import run

run(weights='data/models/yoloTrained.pt', source="test-video.mp4", device="cpu", project="data")