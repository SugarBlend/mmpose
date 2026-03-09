### Launch label studio server
```shell
cd project\label_studio
.\launch-label-studio.ps1
```
### Fetch annotations
Fetch tasks from ls and convert them to coco format (update secrets .env file firstly):
```shell
cd project\label_studio
python .\tasks2json.py
```
Labeling detections manually by using yolo model:
```shell
cd project\label_studio
python .\annotate_person.py
```
And same with pose, which based on detector annotation previously:
```shell
cd project\label_studio
python .\annotate_pose.py
```

After that your can transform coco json to label studio format for create tasks on server
```shell
cd project\label_studio
python .\coco2label_studio.py
```

Add ml pipeline for label studio server (task Pose estimation, but previously you need to update secrets file .env):
```shell
docker-compose -f .\project\label_studio\ml_backend\docker-compose.yml up --build
```