# Python version to use
FROM python:2.7

# Defining working directory and adding source code
WORKDIR /rest-service-api
COPY controllers ./controllers
COPY data ./data
COPY requirements.txt ./
COPY run.sh ./

# Install dependencies
RUN pip install -r requirements.txt

#Start app
EXPOSE 5000
ENTRYPOINT ["/rest-service-api/run.sh"]

# CMD ["python", "./controllers/main.py", "--host=0.0.0.0"]




