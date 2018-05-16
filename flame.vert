//
//  nBody Vertex shader
//

void main()
{
   //  Remember the color
   gl_FrontColor = gl_Color;
   //gl_FrontColor = vec4(1.0,1.0,1.0,1.0);
   //  Defer all transformations to geometry shader
   gl_Position   = gl_Vertex;
}
