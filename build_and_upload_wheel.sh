#!/bin/bash

python3 -m build --wheel
twine upload dist/*

