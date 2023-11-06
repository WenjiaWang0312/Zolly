#!/bin/bash

find . -type d -name "__pycache__" -print -exec rm -r {} \;