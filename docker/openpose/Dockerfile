FROM stanfordnmbl/openpose-gpu:11.3.1
COPY openpose /openpose
COPY defaultOpenCapSettings.json /openpose
RUN pip3 install --upgrade pip
RUN pip3 install -r /openpose/requirements.txt
CMD python3.6 /openpose/loop_openpose.py
