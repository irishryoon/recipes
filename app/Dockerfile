FROM python:3.7

EXPOSE 8501
# set working directory
WORKDIR /app

# copy files from host to current location
COPY requirements.txt ./requirements.txt

# install dependencies
RUN pip3 install -r requirements.txt

COPY . .

# run the command
CMD streamlit run app.py
