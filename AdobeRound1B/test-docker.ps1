# Docker Build and Test Script for Adobe Hackathon

## Build the container
Write-Host "Building Docker container..." -ForegroundColor Green
docker build -t adobe-intelligent-analyzer .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Docker build successful!" -ForegroundColor Green
} else {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

## Test the container
Write-Host "Testing container..." -ForegroundColor Green
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" adobe-intelligent-analyzer

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Container test successful!" -ForegroundColor Green
} else {
    Write-Host "❌ Container test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Adobe Hackathon submission is ready!" -ForegroundColor Cyan
