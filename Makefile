#Makefile for univerSVM software (c) 2005 

CXX= g++
CFLAGS= -O3 
SVQPSRC= svqp2/svqp2.cpp svqp2/vector.c svqp2/messages.c
USVM= universvm
LSVM2BIN= libsvm2bin
USVMSRC= usvm.cpp
LSVM2BINSRC= libsvm2bin.cpp


universvm:
	$(CXX) $(CFLAGS) -o $(USVM) $(USVMSRC) $(SVQPSRC) $(ONLINESRC) -lm

libsvm2bin:
	$(CXX) $(CFLAGS) -o $(LSVM2BIN) $(LSVM2BINSRC) $(SVQPSRC) $(ONLINESRC) -lm

all:
	$(CXX) $(CFLAGS) -o $(USVM) $(USVMSRC) $(SVQPSRC) $(ONLINESRC) -lm
	$(CXX) $(CFLAGS) -o $(LSVM2BIN) $(LSVM2BINSRC) $(SVQPSRC) $(ONLINESRC) -lm
