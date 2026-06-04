FROM stanfordnmbl/opensim-python:4.3

# We used to do this locally until 12/2/23...but there isn't a Release file for ubuntu 20.04.
# RUN add-apt-repository ppa:jonathonf/ffmpeg-4 ; apt update
# RUN apt-get install ffmpeg -y --fix-missing

# Combine apt commands into a single layer to reduce image size
RUN apt update && apt install --no-install-recommends -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/
COPY requirements.txt /workspace/

# Upgrade pip and install Python dependencies in one layer to reduce intermediate image size
RUN pip3 install --upgrade pip --no-cache-dir && pip3 install -r requirements.txt --no-cache-dir

COPY . /workspace/

ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

CMD python3.8 app.py
