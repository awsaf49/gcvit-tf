#!/bin/bash
isort --sl gcvit
black --line-length 80 gcvit
flake8 gcvit