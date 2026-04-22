# GitHub Push Commands
git status
git add .
git commit -m "Update App Runner inference app and model loading"
git pull --rebase origin main
git push origin main

===============
# AWS Snapshot Retrieval Commands
aws s3 cp s3://jac6779-citibike-snapshots-2026/citibike_snapshots/2026/03/2X/ ./snapshots --recursive

===============
Docker Push Commands to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 528757830050.dkr.ecr.us-east-1.amazonaws.com

docker build -t citi-bike-mlops-api .

docker tag citi-bike-mlops-api:latest 528757830050.dkr.ecr.us-east-1.amazonaws.com/citi-bike-mlops-api:v9

docker push 528757830050.dkr.ecr.us-east-1.amazonaws.com/citi-bike-mlops-api:v9