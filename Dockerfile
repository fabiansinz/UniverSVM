FROM gcc:4.9

WORKDIR /usvm

COPY . /usvm/

RUN make all


