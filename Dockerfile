FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN rm -rf /opt/conda/pip.conf /usr/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

ADD ./resources/nltk_data /root/nltk_data
WORKDIR /app
ADD ./requirements.txt requirements.txt
RUN pip install -r ./requirements.txt

ENTRYPOINT ["python", "server.py"]
# ENTRYPOINT ["gunicorn", "-w", "4", "server:app"]