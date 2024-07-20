
PROJECT_NAME="fraud-detection-10062930"
APP_NAME="fraud-web-app"


docker build -t eu.gcr.io/$PROJECT_NAME/$APP_NAME:v1 . --file app/Dockerfile --platform linux/amd64

docker push eu.gcr.io/$PROJECT_NAME/$APP_NAME:v1