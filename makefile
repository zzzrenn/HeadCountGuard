# run sample script
run-sample:
	python src/main.py --config configs/yolo_config.yaml

# run tests
run-tests:
	pytest -v

run-app:
	python src/app.py
