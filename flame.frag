//
//  nBody fragment shader
//
uniform sampler2D star;
//in vec2 texcoord;

void main()
{
   gl_FragColor = texture2D(star,gl_TexCoord[0].st) * gl_Color / 10.0;
   //gl_FragColor = texture2D(star,texcoord) * gl_Color;
}
