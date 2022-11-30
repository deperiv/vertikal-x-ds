FROM python:3.10.5

RUN pip3 install --upgrade pip

RUN mkdir /data_extraction

COPY requirements.txt /data_extraction

WORKDIR /data_extraction

RUN pip3 install -r requirements.txt

COPY . /data_extraction

EXPOSE 6000

CMD ["uvicorn", "data_extraction:app", "--host", "0.0.0.0", "--port", "6000"]