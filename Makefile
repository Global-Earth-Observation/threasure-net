db-black:
	-@black ./src/deep-tree/db/*py

db-isort:
	-@isort ./src/deep-tree/db/*py

db-pylint:
	-@pylint --exit-zero --ignore-patterns "flycheck_*" ./src/deep-tree/db/*py

db-mypy:
	-@PYTHONPATH=. mypy ./src/deep-tree/db/*py

db-ruff:
	-@ruff check ./src/deep-tree/db/*py

db-lint: db-pylint db-ruff db-mypy

db-fix: db-black db-isort

s2sr-black:
	-@black ./src/deep-tree/sentinel2_superresolution/*py

s2sr-isort:
	-@isort ./src/deep-tree/sentinel2_superresolution/*py

s2sr-pylint:
	-@pylint --exit-zero --ignore-patterns "flycheck_*" ./src/deep-tree/sentinel2_superresolution/*py

s2sr-mypy:
	-@PYTHONPATH=. mypy ./src/deep-tree/sentinel2_superresolution/*py

s2sr-ruff:
	-@ruff check ./src/deep-tree/sentinel2_superresolution/*py

s2sr-tests:
	-@PYTHONPATH=./src/deep-tree/sentinel2_superresolution:./src/deep-tree/db pytest --cov=./src/deep-tree/sentinel2_superresolution ./src/deep-tree/sentinel2_superresolution/test_core.py -q -p no:warnings --cov-report=term

s2sr-lint: s2sr-pylint s2sr-ruff s2sr-mypy

s2sr-fix: s2sr-black s2sr-isort

thght-black:
	-@black ./src/deep-tree/tree_height_superresolution/thght_superresolution/*py ./src/deep-tree/tree_height_superresolution/tests/*py ./src/deep-tree/tree_height_superresolution/bin/*py ./src/deep-tree/tree_height_superresolution/scripts/standardization_metrics.py

thght-isort:
	-@isort ./src/deep-tree/tree_height_superresolution/thght_superresolution/*py ./src/deep-tree/tree_height_superresolution/tests/*py ./src/deep-tree/tree_height_superresolution/bin/*py ./src/deep-tree/tree_height_superresolution/scripts/standardization_metrics.py

thght-pylint:
	-@PYTHONPATH=./src/deep-tree pylint --exit-zero --ignore-patterns "flycheck_*" ./src/deep-tree/tree_height_superresolution/thght_superresolution/*py ./src/deep-tree/tree_height_superresolution/tests/*py ./src/deep-tree/tree_height_superresolution/bin/*py ./src/deep-tree/tree_height_superresolution/scripts/standardization_metrics.py

thght-mypy:
	-@PYTHONPATH=./src/deep-tree/db:./src/deep-tree/tree_height_superresolution mypy ./src/deep-tree/tree_height_superresolution/thght_superresolution/

thght-ruffix:
	-@ruff check ./src/deep-tree/tree_height_superresolution/thght_superresolution ./src/deep-tree/tree_height_superresolution/tests ./src/deep-tree/tree_height_superresolution/bin --fix --output-format pylint

thght-ruff:
	-@ruff check ./src/deep-tree/tree_height_superresolution/thght_superresolution ./src/deep-tree/tree_height_superresolution/tests ./src/deep-tree/tree_height_superresolution/bin

thght-test-dataset:
	-@PYTHONPATH=./src/deep-tree:./src/deep-tree/tree_height_superresolution:./src/deep-tree/db:./src/deep-tree/tree_height_superresolution/thght_superresolution/torchsisr pytest --cov=./src/deep-tree/tree_height_superresolution ./src/deep-tree/tree_height_superresolution/tests/dataset_test.py --cov-report=term -q -p no:warnings

thght-tests:
	-@PYTHONPATH=./src/deep-tree:./src/deep-tree/tree_height_superresolution:./src/deep-tree/db:./src/deep-tree/tree_height_superresolution/thght_superresolution/torchsisr pytest --cov=./src/deep-tree/tree_height_superresolution ./src/deep-tree/tree_height_superresolution/tests/*py --cov-report=term -q -p no:warnings

thght-lint: thght-pylint thght-ruff thght-mypy

thght-fix: thght-isort thght-ruffix

lhd-black:
	-@black ./src/deep-tree/lidarhd/commons/*py ./src/deep-tree/lidarhd/download/*py ./src/deep-tree/lidarhd/metrics/*py ./src/deep-tree/lidarhd/*py

lhd-isort:
	-@isort ./src/deep-tree/lidarhd/commons/*py ./src/deep-tree/lidarhd/download/*py ./src/deep-tree/lidarhd/metrics/*py ./src/deep-tree/lidarhd/*py

lhd-pylint:
	-@PYTHONPATH=./src/deep-tree/lidarhd pylint --exit-zero --ignore-patterns "flycheck_*" ./src/deep-tree/lidarhd/commons/*py ./src/deep-tree/lidarhd/download/*py ./src/deep-tree/lidarhd/metrics/*py ./src/deep-tree/lidarhd/*py

lhd-mypy:
	-@PYTHONPATH=. mypy ./src/deep-tree/lidarhd/download/*py ./src/deep-tree/lidarhd/commons/*py ./src/deep-tree/lidarhd/metrics/*.py ./src/deep-tree/lidarhd/*py

lhd-tests:
	-@PYTHONPATH=./src/deep-tree/lidarhd/ pytest --cov=./src/deep-tree/lidarhd ./src/deep-tree/lidarhd/lidar_hd_test.py -q -p no:warnings

lhd-lint: lhd-pylint lhd-mypy

lhd-fix: lhd-isort lhd-ruffix

lint-all: db-lint s2sr-lint thght-lint lhd-lint

test-all: s2sr-tests lhd-tests thght-tests
