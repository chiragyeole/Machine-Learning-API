#!/bin/sh
python ./controllers/create_db.py
export FLASK_APP=./controllers/main.py
flask run -h 0.0.0.0
