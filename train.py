from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov5su.pt")

    train_results = model.train(
        data="Dataset/dataset.yaml",  # Path to dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="cuda",
        optimizer= "Adam"
    )