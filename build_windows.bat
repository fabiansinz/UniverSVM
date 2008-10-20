rem Makefile for univerSVM software (c) 2005 for .NET 2005 console
rem by Matteo Roffilli - roffilli@csr.unibo.it
rem
rem tested also on Windows Vista + .NET 2008 console

del *.obj
del universvm.exe
cl /EHsc /D_CRT_SECURE_NO_DEPRECATE /TP /Ox /W3 /c svqp2/vector.c
cl /EHsc /D_CRT_SECURE_NO_DEPRECATE /TP /Ox /W3 /c svqp2/messages.c
cl /EHsc /D_CRT_SECURE_NO_DEPRECATE /TP /Ox /W3 /c svqp2/svqp2.cpp
cl /EHsc /D_WIN32_ /D_CRT_SECURE_NO_DEPRECATE /Ox /W3 /Feuniversvm.exe vector.obj messages.obj svqp2.obj usvm.cpp
