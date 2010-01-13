#Makefile for univerSVM software (c) 2005 

CXX= g++
MEXFLAGS= -DMATLAB_MEX_FILE -DMEX
USEMEXFLAGS= 
CFLAGS= -O3 
SVQPSRC= svqp2/svqp2.cpp svqp2/vector.c svqp2/messages.c
USVM= universvm
USVM64= universvm64
LSVM2BIN= libsvm2bin
BIN2LSVM= bin2libsvm
USVMSRC= usvm.cpp
LSVM2BINSRC= libsvm2bin.cpp
BIN2LSVMSRC= bin2libsvm.cpp
MEX = ./mex


svqp: $(SVQPSRC)
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -fPIC -o svqp2/vector.o -c svqp2/vector.c
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -fPIC -o svqp2/messages.o -c svqp2/messages.c
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -fPIC -o svqp2/svqp2.o -c svqp2/svqp2.cpp

mex: $(USVMSRC) $(SVQPSRC)
	make "USEMEXFLAGS=$(MEXFLAGS)" svqp
	$(MEX) $(MEXFLAGS) $(USVMSRC)  svqp2/svqp2.o svqp2/messages.o svqp2/vector.o

universvm: $(USVMSRC)
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -o $(USVM) $(USVMSRC) $(SVQPSRC) $(ONLINESRC) -lm

universvm64: $(USVMSRC)
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -o $(USVM64) $(USVMSRC) $(SVQPSRC) $(ONLINESRC) -lm

libsvm2bin: $(LSVM2BINSRC)
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -o $(LSVM2BIN) $(LSVM2BINSRC) $(SVQPSRC) $(ONLINESRC) -lm

bin2libsvm: $(BIN2LSVMSRC) 
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -o $(BIN2LSVM) $(BIN2LSVMSRC) $(SVQPSRC) $(ONLINESRC) -lm

all: $(USVMSRC) $(LSVM2BINSRC)
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -o $(USVM) $(USVMSRC) $(SVQPSRC) $(ONLINESRC) -lm
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -o $(LSVM2BIN) $(LSVM2BINSRC) $(SVQPSRC) $(ONLINESRC) -lm
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -o $(BIN2LSVM) $(BIN2LSVMSRC) $(SVQPSRC) $(ONLINESRC) -lm
