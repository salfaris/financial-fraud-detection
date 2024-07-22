docker build -t dev:latest . --file app/Dockerfile

docker run --rm -p 8080:8080 dev:latest