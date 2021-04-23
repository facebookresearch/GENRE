# How to use:
#   docker build --tag genre:latest .
#   docker run --rm -it genre:latest /bin/bash
#   docker run --rm -it -v $(pwd)/tests:/GENRE/genre/tests genre:latest /bin/bash
#   pytest genre/tests
FROM python:3.8

WORKDIR /GENRE/

RUN apt-get update && \
    apt-get install axel -y

RUN mkdir data && \
    axel -n 20 http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl -o data

RUN mkdir models && \
    axel -n 20 http://dl.fbaipublicfiles.com/GENRE/fairseq_wikipage_retrieval.tar.gz && \
    tar -xvf fairseq_wikipage_retrieval.tar.gz --directory models && \
    rm fairseq_wikipage_retrieval.tar.gz

# Install PyTorch
RUN pip install torch --no-cache-dir

# Install dependencies
RUN pip install pytest requests --no-cache-dir

# Install fairseq
RUN git clone -b fixing_prefix_allowed_tokens_fn --single-branch https://github.com/nicola-decao/fairseq.git
RUN pip install -e ./fairseq

# Install genre
COPY . genre
RUN pip install -e ./genre