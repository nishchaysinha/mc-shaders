#version 120

#define gbuffers_shadows
#include "shaders.settings"

#ifdef Shadows
varying vec4 texcoord;
uniform sampler2D texture;
uniform int blockEntityId;
uniform int entityId;
#endif

void main() {

#ifdef Shadows
	vec4 color = texture2D(texture, texcoord.xy);
	if(texcoord.z > 0.9)color.rgb = vec3(1.0, 1.0, 1.0);	//water shadows color
	if(texcoord.w > 0.9)color = vec4(0.0); 					//disable shadows on entities defined in vertex shadows
	if(blockEntityId == 10089.0 || entityId == 11000.0)color *= 0.0;	//remove beacon beam shadows, 10089 is actually the id of all emissive blocks but only the beam should be a block entity, also remove lightning strike shadow.
	gl_FragData[0] = color;
#else
	gl_FragData[0] = vec4(0.0);
#endif	
}