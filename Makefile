#  MinGW
#ifeq "$(OS)" "Windows_NT"
#SFLG= -O3 -Wall
#LIBS=-lglut32cu -lglu32 -lopengl32
#CLEAN=del *.exe *.o *.a
#else
#  OSX
ifeq ("$(shell uname)","Darwin")
SFLG=-Wno-deprecated-declarations $(shell sdl2-config --cflags)
GLIBS=-framework OpenGL $(shell sdl2-config --libs)
CLEAN=rm -rf run Test main *.o *.a *.dSYM
#  Linux/Unix/Solaris
else
SFLG=$(shell sdl2-config --cflags) -DGL_GLEXT_PROTOTYPES
GLIBS=-lGLU -lGL -lm $(shell sdl2-config --libs)
CLEAN=rm -rf run Test main *.o *.a
endif
#endif
CFLG=-g -O3 -Wall #-std=c++11
#CFLG=-g -Wall #-std=c++11
ARCH=-gencode arch=compute_75,code=sm_75 \
		 -gencode arch=compute_86,code=sm_86

all:main


objects.o:objects.cpp stdGL.h
	g++ -c $(CFLG) $(SFLG) $<

shader.o:shader.cpp shader.h
	g++ -c $(CFLG) $(SFLG) $<

#step.o:step.cu step.h
#	nvcc -c $< -Xcompiler "$(CFLG) -fopenmp"

main.o:main.cu step.cu shader.h objects.h stdGL.h
	nvcc -lineinfo $(ARCH) -ccbin g++ -c -Xcompiler "$(CFLG) $(SFLG) -fPIC -fopenmp" $<

#  link
main:main.o shader.o objects.o #step.o
	nvcc -lineinfo $(ARCH) -g -O3 -o $@ $^ -Xlinker "$(GLIBS) -lgomp -fPIC"


#  Clean
clean:
	$(CLEAN)
