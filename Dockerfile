# Base Image
FROM  python:3.9


# WOrking Directory
WORKDIR /app

# copy
COPY . /app

# Run commads
RUN pip install -r requirements.txt 

# port 
EXPOSE 5000

# Command to run the application
CMD [ "python","./app.py" ]