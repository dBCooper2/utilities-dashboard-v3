.PHONY: build up down logs shell db clean test

# Build all containers
build:
	docker-compose build

# Start all containers
up:
	docker-compose up -d

# Start all containers and show logs
start:
	docker-compose up

# Stop all containers
down:
	docker-compose down

# Show logs
logs:
	docker-compose logs -f

# Open shell in backend container
shell:
	docker-compose exec backend-prod /bin/bash

# Connect to database
db:
	docker-compose exec timescaledb psql -U postgres -d weather_db

# Remove volumes and orphaned containers
clean:
	docker-compose down -v --remove-orphans

# Run tests
test:
	docker-compose run --rm backend-prod python tests/test_florida_region.py

# Help
help:
	@echo "Available commands:"
	@echo "  make build  - Build all containers"
	@echo "  make up     - Start all containers in detached mode"
	@echo "  make start  - Start all containers and show logs"
	@echo "  make down   - Stop all containers"
	@echo "  make logs   - Show logs"
	@echo "  make shell  - Open shell in backend container" 
	@echo "  make db     - Connect to database"
	@echo "  make clean  - Remove volumes and orphaned containers"
	@echo "  make test   - Run tests" 