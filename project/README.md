## Setting virtual environment
```powershell
pip install -r project/requirements.txt
```

## Create person detections
```powershell
python project/annotations/detection_labels.py --yolo-version yolo12x.pt --files project/dataset/images --output-json project/dataset/coco_annotations.json
```

## Convert COCO format to Label Studio
```powershell
project/annotations/converter.ps1
```

## Launch in Label Studio:
```powershell
project/annotations/launch-label-studio.ps1
```
