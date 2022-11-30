FROM python:3.10.5

RUN pip3 install --upgrade pip

RUN mkdir /sentiment_analysis

COPY requirements.txt /sentiment_analysis

WORKDIR /sentiment_analysis

RUN curl https://sh.rustup.rs -sSf | sh export PATH=$PATH:~/.cargo/bin

RUN pip3 install -r requirements.txt

COPY . /sentiment_analysis

EXPOSE 6001

CMD ["uvicorn", "sentiment_analysis:app", "--host", "0.0.0.0", "--port", "6001"]