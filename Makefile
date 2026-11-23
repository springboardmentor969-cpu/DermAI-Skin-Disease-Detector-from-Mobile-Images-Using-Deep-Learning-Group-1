.PHONY: test lint dev down logs

test:
	cd backend && pytest tests/ -v

lint:
	cd backend && flake8 app/ && cd ../frontend && npm run lint

dev:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f
