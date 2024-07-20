docker build -t dev:latest . --file app/Dockerfile

docker run -p 8080:8080 dev:latest