#include "stdGL.h"

#ifndef SHADER_H
#define SHADER_H

// this function stolen from 4229 class examples
char* ReadText(const char* file);
// this function stolen from 4229 class examples
int CreateShader(GLenum type, char* file);
void CreateShader(int prog, const GLenum type, const char* file);
// this function stolen (mostly) from 4229 class examples
int CreateShaderProg(char* VertFile, char* FragFile);
int CreateShaderProgGeom(char* VertFile, char* GeomFile, char* FragFile);


#endif
