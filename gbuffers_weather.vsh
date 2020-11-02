#version 120

#define gbuffers_weather
#include "shaders.settings"

varying vec4 color;
varying vec2 texcoord;
varying float lmcoord;

#ifdef WeatherAngle
uniform vec3 cameraPosition;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
#endif

void main() {

#ifdef WeatherAngle
	vec4 position = gbufferModelViewInverse * gl_ModelViewMatrix * gl_Vertex;
	float worldpos = position.y + cameraPosition.y;
	bool istopv = worldpos > cameraPosition.y+5.0;
	if (!istopv) position.xz += vec2(3.0,1.0);
	gl_Position = gl_ProjectionMatrix * gbufferModelView * position;
#else
	gl_Position = ftransform();
#endif
	
	texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
	lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).s;
	color = gl_Color;

}
