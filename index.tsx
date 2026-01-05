import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';

// Fix: Declare WebGPU constants to avoid TypeScript errors
declare const GPUBufferUsage: any;
declare const GPUTextureUsage: any;

// --- Linear Algebra Helpers ---
const Mat4 = {
  create: () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  perspective: (out: Float32Array, fovy: number, aspect: number, near: number, far: number) => {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = (far + near) * nf; out[11] = -1;
    out[12] = 0; out[13] = 0; out[14] = (2 * far * near) * nf; out[15] = 0;
    return out;
  },
  lookAt: (out: Float32Array, eye: number[], center: number[], up: number[]) => {
    let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
    let eyex = eye[0], eyey = eye[1], eyez = eye[2];
    let upx = up[0], upy = up[1], upz = up[2];
    let centerx = center[0], centery = center[1], centerz = center[2];

    z0 = eyex - centerx; z1 = eyey - centery; z2 = eyez - centerz;
    len = 1 / Math.hypot(z0, z1, z2);
    z0 *= len; z1 *= len; z2 *= len;

    x0 = upy * z2 - upz * z1; x1 = upz * z0 - upx * z2; x2 = upx * z1 - upy * z0;
    len = Math.hypot(x0, x1, x2);
    if (!len) { x0 = 0; x1 = 0; x2 = 0; } else { len = 1 / len; x0 *= len; x1 *= len; x2 *= len; }

    y0 = z1 * x2 - z2 * x1; y1 = z2 * x0 - z0 * x2; y2 = z0 * x1 - z1 * x0;
    len = Math.hypot(y0, y1, y2);
    if (!len) { y0 = 0; y1 = 0; y2 = 0; } else { len = 1 / len; y0 *= len; y1 *= len; y2 *= len; }

    out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
    out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
    out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
    out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
    out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
    out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
    out[15] = 1;
    return out;
  },
  multiply: (out: Float32Array, a: Float32Array, b: Float32Array) => {
    let a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    let a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    let a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    let a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[1] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[2] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[3] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
    b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
    out[4] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[5] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[6] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[7] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
    b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
    out[8] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[9] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[10] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[11] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
    b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
    out[12] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[13] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[14] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[15] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
    return out;
  }
};

const Vec3 = {
    normalize: (v: number[]) => {
        const len = Math.hypot(v[0], v[1], v[2]);
        if (len > 0) return [v[0]/len, v[1]/len, v[2]/len];
        return [0,0,0];
    }
}

// --- WGSL Shaders ---

const shaderWGSL = `
struct Uniforms {
    viewProjectionMatrix : mat4x4<f32>,
    cameraPosition : vec3<f32>,
    _pad1 : f32,
    sunDirection : vec3<f32>,
    _pad2 : f32,
    planetCenter : vec3<f32>,
    planetRadius : f32,
    atmosphereRadius : f32,
    scatteringStrength : f32,
    mieCoeff : f32,
    rayleighScaleHeight : f32,
    rayleighCoeff : vec3<f32>,
    mieScaleHeight : f32,
    mieG : f32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) worldPos : vec3<f32>,
};

@vertex
fn vs_main(@location(0) position : vec3<f32>) -> VertexOutput {
    var output : VertexOutput;
    let worldPosition = position * uniforms.atmosphereRadius * 1.05 + uniforms.planetCenter;
    output.Position = uniforms.viewProjectionMatrix * vec4<f32>(worldPosition, 1.0);
    output.worldPos = worldPosition;
    return output;
}

fn raySphereIntersect(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let h = b * b - c;
    if (h < 0.0) { return vec2<f32>(-1.0); }
    let sqrtH = sqrt(h);
    return vec2<f32>(-b - sqrtH, -b + sqrtH);
}

@fragment
fn fs_main(@location(0) worldPos : vec3<f32>) -> @location(0) vec4<f32> {
    let rayOrigin = uniforms.cameraPosition;
    let rayDir = normalize(worldPos - rayOrigin);
    
    let dstToAtmosphere = raySphereIntersect(rayOrigin, rayDir, uniforms.planetCenter, uniforms.atmosphereRadius);
    if (dstToAtmosphere.y < 0.0) { discard; }
    
    var dstToAtmosphereNear = max(0.0, dstToAtmosphere.x);
    var dstToAtmosphereFar = dstToAtmosphere.y;
    
    let dstToPlanet = raySphereIntersect(rayOrigin, rayDir, uniforms.planetCenter, uniforms.planetRadius);
    var dstToSurface = dstToAtmosphereFar;
    var hitsGround = false;
    
    if (dstToPlanet.x > 0.0) {
        dstToSurface = dstToPlanet.x;
        hitsGround = true;
    }

    let numSteps = 12; 
    let stepSize = (dstToSurface - dstToAtmosphereNear) / f32(numSteps);
    var currentStepDistance = dstToAtmosphereNear;
    
    var totalRayleigh = vec3<f32>(0.0);
    var totalMie = vec3<f32>(0.0);
    var opticalDepthRayleigh = 0.0;
    var opticalDepthMie = 0.0;
    
    let mu = dot(rayDir, uniforms.sunDirection);
    let phaseRayleigh = 3.0 / (16.0 * 3.14159) * (1.0 + mu * mu);
    let g = uniforms.mieG;
    let phaseMie = 3.0 / (8.0 * 3.14159) * ((1.0 - g * g) * (1.0 + mu * mu)) / ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
    
    for (var i = 0; i < numSteps; i++) {
        let samplePos = rayOrigin + rayDir * (currentStepDistance + stepSize * 0.5);
        let height = length(samplePos - uniforms.planetCenter) - uniforms.planetRadius;
        
        if (height < 0.0) { break; } 

        let densityRayleigh = exp(-height / uniforms.rayleighScaleHeight) * stepSize;
        let densityMie = exp(-height / uniforms.mieScaleHeight) * stepSize;
        
        opticalDepthRayleigh += densityRayleigh;
        opticalDepthMie += densityMie;
        
        let lightRayOrigin = samplePos;
        let lightDstToAtmosphere = raySphereIntersect(lightRayOrigin, uniforms.sunDirection, uniforms.planetCenter, uniforms.atmosphereRadius);
        let lightDstToPlanet = raySphereIntersect(lightRayOrigin, uniforms.sunDirection, uniforms.planetCenter, uniforms.planetRadius);
        
        if (lightDstToPlanet.y < 0.0) {
            let numLightSteps = 4;
            let lightStepSize = lightDstToAtmosphere.y / f32(numLightSteps);
            var lightOpticalDepthRayleigh = 0.0;
            var lightOpticalDepthMie = 0.0;
            
            for (var j = 0; j < numLightSteps; j++) {
                 let lightSamplePos = lightRayOrigin + uniforms.sunDirection * (f32(j) + 0.5) * lightStepSize;
                 let lightHeight = length(lightSamplePos - uniforms.planetCenter) - uniforms.planetRadius;
                 if (lightHeight > 0.0) {
                     lightOpticalDepthRayleigh += exp(-lightHeight / uniforms.rayleighScaleHeight) * lightStepSize;
                     lightOpticalDepthMie += exp(-lightHeight / uniforms.mieScaleHeight) * lightStepSize;
                 }
            }
            
            let totalOpticalDepthRayleigh = opticalDepthRayleigh + lightOpticalDepthRayleigh;
            let totalOpticalDepthMie = opticalDepthMie + lightOpticalDepthMie;
            let transmittance = exp(-(uniforms.rayleighCoeff * totalOpticalDepthRayleigh + uniforms.mieCoeff * totalOpticalDepthMie));
            
            totalRayleigh += densityRayleigh * transmittance;
            totalMie += densityMie * transmittance;
        }
        
        currentStepDistance += stepSize;
    }

    let sunIntensity = 20.0;
    var color = (totalRayleigh * uniforms.rayleighCoeff * phaseRayleigh + totalMie * uniforms.mieCoeff * phaseMie) * uniforms.scatteringStrength * sunIntensity;
    
    if (hitsGround) {
        let groundTransmittance = exp(-(uniforms.rayleighCoeff * opticalDepthRayleigh + uniforms.mieCoeff * opticalDepthMie));
        let hitPos = rayOrigin + rayDir * dstToSurface;
        let normal = normalize(hitPos - uniforms.planetCenter);
        let NdotL = max(dot(normal, uniforms.sunDirection), 0.0);
        let groundAlbedo = vec3<f32>(0.05, 0.1, 0.25); 
        color += groundAlbedo * NdotL * groundTransmittance * sunIntensity * 0.5;
    }

    // output HDR color directly (Tone Mapping moved to Composite pass)
    return vec4<f32>(color, 1.0);
}

// --- Post Processing Shaders ---

struct BlurUniforms {
    direction: vec2<f32>,
    resolution: vec2<f32>,
    threshold: f32, // Used only in first pass to extract bright spots
};

@group(0) @binding(0) var sceneTexture : texture_2d<f32>;
@group(0) @binding(1) var sceneSampler : sampler;
@group(0) @binding(2) var<uniform> blurConfig : BlurUniforms;

struct QuadVertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_quad(@builtin(vertex_index) vertexIndex : u32) -> QuadVertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    var output : QuadVertexOutput;
    output.Position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = pos[vertexIndex] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y; // Flip Y for texture coords
    return output;
}

@fragment
fn fs_blur(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let texSize = blurConfig.resolution;
    let dir = blurConfig.direction;
    
    // 9-tap Gaussian
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    
    var color = vec3<f32>(0.0);
    
    // Center sample
    var centerSample = textureSample(sceneTexture, sceneSampler, uv).rgb;
    
    // Apply Threshold (Bright Pass) only if threshold > 0 (Horizontal Pass)
    if (blurConfig.threshold > 0.0) {
        centerSample = max(centerSample - vec3<f32>(blurConfig.threshold), vec3<f32>(0.0));
    }
    
    color += centerSample * weights[0];
    
    for (var i = 1; i < 5; i++) {
        let offset = vec2<f32>(f32(i)) * dir / texSize * 2.5; // * Spread factor
        
        var sample1 = textureSample(sceneTexture, sceneSampler, uv + offset).rgb;
        var sample2 = textureSample(sceneTexture, sceneSampler, uv - offset).rgb;
        
        if (blurConfig.threshold > 0.0) {
             sample1 = max(sample1 - vec3<f32>(blurConfig.threshold), vec3<f32>(0.0));
             sample2 = max(sample2 - vec3<f32>(blurConfig.threshold), vec3<f32>(0.0));
        }
        
        color += (sample1 + sample2) * weights[i];
    }
    
    return vec4<f32>(color, 1.0);
}

struct CompositeUniforms {
    bloomStrength: f32,
}

@group(0) @binding(0) var tScene : texture_2d<f32>;
@group(0) @binding(1) var tBloom : texture_2d<f32>;
@group(0) @binding(2) var tSampler : sampler;
@group(0) @binding(3) var<uniform> compConfig : CompositeUniforms;

@fragment
fn fs_composite(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    var sceneColor = textureSample(tScene, tSampler, uv).rgb;
    var bloomColor = textureSample(tBloom, tSampler, uv).rgb;
    
    var color = sceneColor + bloomColor * compConfig.bloomStrength;
    
    // Tone Mapping (Reinhardt Extended or ACES)
    // Simple Reinhardt
    color = color / (color + vec3<f32>(1.0));
    
    // Gamma Correction
    color = pow(color, vec3<f32>(1.0/2.2));
    
    return vec4<f32>(color, 1.0);
}
`;

function createSphereGeometry(radius: number, widthSegments = 64, heightSegments = 32) {
    const positions = [];
    const indices = [];
    
    for(let y = 0; y <= heightSegments; y++) {
        for(let x = 0; x <= widthSegments; x++) {
            const u = x / widthSegments;
            const v = y / heightSegments;
            const theta = u * Math.PI * 2;
            const phi = v * Math.PI;
            
            const px = Math.sin(phi) * Math.cos(theta);
            const py = Math.cos(phi);
            const pz = Math.sin(phi) * Math.sin(theta);
            
            positions.push(px * radius, py * radius, pz * radius);
        }
    }
    
    for (let y = 0; y < heightSegments; y++) {
        for (let x = 0; x < widthSegments; x++) {
            const v1 = y * (widthSegments + 1) + x;
            const v2 = v1 + widthSegments + 1;
            
            indices.push(v1, v2, v1 + 1);
            indices.push(v2, v2 + 1, v1 + 1);
        }
    }
    
    return { 
        positions: new Float32Array(positions), 
        indices: new Uint16Array(indices), 
        count: indices.length 
    };
}

const App = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [error, setError] = useState<string | null>(null);
    const [sunAngle, setSunAngle] = useState(0.42); // Late afternoon look
    const [scatteringStrength, setScatteringStrength] = useState(1.5);
    const [bloomStrength, setBloomStrength] = useState(0.8);
    const [zoom, setZoom] = useState(2400.0);
    
    // Use ref to pass latest state to render loop without re-binding
    const paramsRef = useRef({
        sunAngle: 0.42,
        scatteringStrength: 1.5,
        bloomStrength: 0.8,
        zoom: 2400.0,
        rotationX: 0.1,
        rotationY: 0.5,
    });

    useEffect(() => {
        paramsRef.current.sunAngle = sunAngle;
        paramsRef.current.scatteringStrength = scatteringStrength;
        paramsRef.current.bloomStrength = bloomStrength;
        paramsRef.current.zoom = zoom;
    }, [sunAngle, scatteringStrength, bloomStrength, zoom]);

    useEffect(() => {
        const initWebGPU = async () => {
            const gpu = (navigator as any).gpu;
            
            if (!gpu) {
                setError("WebGPU is not supported in this browser.");
                return;
            }

            const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
            if (!adapter) {
                setError("No suitable WebGPU adapter found.");
                return;
            }

            const device = await adapter.requestDevice();
            const canvas = canvasRef.current;
            if (!canvas) return;

            const context = canvas.getContext('webgpu') as any;
            const presentationFormat = gpu.getPreferredCanvasFormat();
            
            if (!context) {
                setError("Could not get WebGPU context.");
                return;
            }

            context.configure({
                device,
                format: presentationFormat,
                alphaMode: 'premultiplied',
            });

            // 1. Create Scene Resources
            const sphere = createSphereGeometry(1.0);
            
            const vertexBuffer = device.createBuffer({
                size: sphere.positions.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(vertexBuffer, 0, sphere.positions);

            const indexBuffer = device.createBuffer({
                size: sphere.indices.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(indexBuffer, 0, sphere.indices);

            const uniformBufferSize = 256; 
            const uniformBuffer = device.createBuffer({
                size: uniformBufferSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            const shaderModule = device.createShaderModule({ code: shaderWGSL });
            
            // --- Pipelines ---
            
            // 1. Scene Pipeline (Renders Atmosphere)
            const scenePipeline = device.createRenderPipeline({
                layout: 'auto',
                vertex: {
                    module: shaderModule,
                    entryPoint: 'vs_main',
                    buffers: [{
                        arrayStride: 12,
                        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
                    }],
                },
                fragment: {
                    module: shaderModule,
                    entryPoint: 'fs_main',
                    targets: [{ format: 'rgba16float' }], // Write to HDR float texture
                },
                primitive: { topology: 'triangle-list', cullMode: 'front' },
                depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' },
            });

            // 2. Blur Pipeline
            const blurPipeline = device.createRenderPipeline({
                layout: 'auto',
                vertex: { module: shaderModule, entryPoint: 'vs_quad' },
                fragment: { 
                    module: shaderModule, 
                    entryPoint: 'fs_blur', 
                    targets: [{ format: 'rgba16float' }] 
                },
                primitive: { topology: 'triangle-list' },
            });

            // 3. Composite Pipeline
            const compositePipeline = device.createRenderPipeline({
                layout: 'auto',
                vertex: { module: shaderModule, entryPoint: 'vs_quad' },
                fragment: { 
                    module: shaderModule, 
                    entryPoint: 'fs_composite', 
                    targets: [{ format: presentationFormat }] 
                },
                primitive: { topology: 'triangle-list' },
            });

            // --- Samplers ---
            const linearSampler = device.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
            });

            // --- Uniforms for Post Processing ---
            const blurHUniformBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            const blurVUniformBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            const compositeUniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });


            const sceneBindGroup = device.createBindGroup({
                layout: scenePipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
            });

            // --- Textures (created on resize) ---
            let depthTexture: any = null;
            let sceneTexture: any = null;
            let blurTexture1: any = null;
            let blurTexture2: any = null;
            
            let blurHBindGroup: any = null;
            let blurVBindGroup: any = null;
            let compositeBindGroup: any = null;

            // Input Handling
            let isDragging = false;
            let lastX = 0;
            let lastY = 0;
            
            const onMouseDown = (e: MouseEvent) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; };
            const onMouseUp = () => { isDragging = false; };
            const onMouseMove = (e: MouseEvent) => {
                if(!isDragging) return;
                const dx = e.clientX - lastX;
                const dy = e.clientY - lastY;
                lastX = e.clientX;
                lastY = e.clientY;
                paramsRef.current.rotationY -= dx * 0.005;
                paramsRef.current.rotationX = Math.max(-1.5, Math.min(1.5, paramsRef.current.rotationX - dy * 0.005));
            };
            const onWheel = (e: WheelEvent) => {
                 setZoom(z => Math.max(1100, Math.min(10000, z + e.deltaY)));
            }

            canvas.addEventListener('mousedown', onMouseDown);
            window.addEventListener('mouseup', onMouseUp);
            window.addEventListener('mousemove', onMouseMove);
            canvas.addEventListener('wheel', onWheel, { passive: true });

            let frameId = 0;

            const render = () => {
                const width = canvas.clientWidth * window.devicePixelRatio;
                const height = canvas.clientHeight * window.devicePixelRatio;
                
                // Resize handling
                if (canvas.width !== width || canvas.height !== height) {
                    canvas.width = width;
                    canvas.height = height;

                    if (depthTexture) depthTexture.destroy();
                    if (sceneTexture) sceneTexture.destroy();
                    if (blurTexture1) blurTexture1.destroy();
                    if (blurTexture2) blurTexture2.destroy();

                    depthTexture = device.createTexture({
                        size: [width, height],
                        format: 'depth24plus',
                        usage: GPUTextureUsage.RENDER_ATTACHMENT,
                    });
                    
                    // HDR Scene Texture
                    sceneTexture = device.createTexture({
                        size: [width, height],
                        format: 'rgba16float',
                        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                    });

                    // Downscaled blur textures for performance and larger glow radius
                    const blurWidth = Math.floor(width / 2);
                    const blurHeight = Math.floor(height / 2);
                    
                    blurTexture1 = device.createTexture({
                        size: [blurWidth, blurHeight],
                        format: 'rgba16float',
                        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                    });

                    blurTexture2 = device.createTexture({
                        size: [blurWidth, blurHeight],
                        format: 'rgba16float',
                        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                    });

                    // Re-create Bind Groups since textures changed
                    
                    // Blur H (Reads Scene, Writes Blur1)
                    blurHBindGroup = device.createBindGroup({
                        layout: blurPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: sceneTexture.createView() },
                            { binding: 1, resource: linearSampler },
                            { binding: 2, resource: { buffer: blurHUniformBuffer } },
                        ]
                    });

                    // Blur V (Reads Blur1, Writes Blur2)
                    blurVBindGroup = device.createBindGroup({
                        layout: blurPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: blurTexture1.createView() },
                            { binding: 1, resource: linearSampler },
                            { binding: 2, resource: { buffer: blurVUniformBuffer } },
                        ]
                    });

                    // Composite (Reads Scene, Reads Blur2, Writes Screen)
                    compositeBindGroup = device.createBindGroup({
                        layout: compositePipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: sceneTexture.createView() },
                            { binding: 1, resource: blurTexture2.createView() },
                            { binding: 2, resource: linearSampler },
                            { binding: 3, resource: { buffer: compositeUniformBuffer } },
                        ]
                    });
                }

                const { sunAngle, scatteringStrength, bloomStrength, zoom, rotationX, rotationY } = paramsRef.current;
                
                // Update Simulation Uniforms
                const aspect = width / height;
                const projMat = Mat4.create();
                Mat4.perspective(projMat, Math.PI / 4, aspect, 10.0, 50000.0);
                
                const camX = Math.sin(rotationY) * Math.cos(rotationX) * zoom;
                const camY = Math.sin(rotationX) * zoom;
                const camZ = Math.cos(rotationY) * Math.cos(rotationX) * zoom;
                const camPos = [camX, camY, camZ];
                
                const viewMat = Mat4.create();
                Mat4.lookAt(viewMat, camPos, [0,0,0], [0,1,0]);
                const viewProj = Mat4.create();
                Mat4.multiply(viewProj, projMat, viewMat);

                const sx = Math.cos(sunAngle * Math.PI * 2);
                const sy = Math.sin(sunAngle * Math.PI * 2) * 0.4;
                const sz = Math.sin(sunAngle * Math.PI * 2) * 0.5;
                const sunDir = Vec3.normalize([sx, sy, sz]);

                const f32Data = new Float32Array(uniformBufferSize / 4);
                f32Data.set(viewProj, 0);
                f32Data.set(camPos, 16);
                f32Data.set(sunDir, 20);
                f32Data.set([0,0,0], 24);
                const planetRadius = 1000.0;
                f32Data[27] = planetRadius;
                f32Data[28] = planetRadius * 1.15;
                f32Data[29] = scatteringStrength;
                f32Data[30] = 21e-6; 
                f32Data[31] = 1.25 * 5.0; 
                f32Data.set([5.8e-6, 13.5e-6, 33.1e-6], 32); 
                f32Data.set([0.1, 0.2, 0.5], 32); 
                f32Data[35] = 0.2 * 5.0;
                f32Data[36] = 0.76;
                device.queue.writeBuffer(uniformBuffer, 0, f32Data);

                // Update Post-Process Uniforms
                // Blur H: Direction (1,0), Resolution (Scene W/H), Threshold (1.0 to extract sun)
                const blurW = Math.floor(width / 2);
                const blurH = Math.floor(height / 2);
                device.queue.writeBuffer(blurHUniformBuffer, 0, new Float32Array([1.0, 0.0, width, height, 1.0, 0.0, 0.0, 0.0]));
                // Blur V: Direction (0,1), Resolution (Blur W/H), Threshold (0.0 already thresholded)
                device.queue.writeBuffer(blurVUniformBuffer, 0, new Float32Array([0.0, 1.0, blurW, blurH, 0.0, 0.0, 0.0, 0.0]));
                // Composite: Bloom Strength
                device.queue.writeBuffer(compositeUniformBuffer, 0, new Float32Array([bloomStrength, 0.0, 0.0, 0.0]));

                const commandEncoder = device.createCommandEncoder();

                // Pass 1: Atmosphere -> SceneTexture
                const pass1 = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: sceneTexture.createView(),
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    }],
                    depthStencilAttachment: {
                        view: depthTexture.createView(),
                        depthClearValue: 1.0,
                        depthLoadOp: 'clear',
                        depthStoreOp: 'store',
                    },
                });
                pass1.setPipeline(scenePipeline);
                pass1.setBindGroup(0, sceneBindGroup);
                pass1.setVertexBuffer(0, vertexBuffer);
                pass1.setIndexBuffer(indexBuffer, 'uint16');
                pass1.drawIndexed(sphere.count);
                pass1.end();

                // Pass 2: Blur H -> BlurTexture1
                const pass2 = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: blurTexture1.createView(),
                        loadOp: 'clear',
                        storeOp: 'store',
                    }],
                });
                pass2.setPipeline(blurPipeline);
                pass2.setBindGroup(0, blurHBindGroup);
                pass2.draw(6);
                pass2.end();

                // Pass 3: Blur V -> BlurTexture2
                const pass3 = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: blurTexture2.createView(),
                        loadOp: 'clear',
                        storeOp: 'store',
                    }],
                });
                pass3.setPipeline(blurPipeline);
                pass3.setBindGroup(0, blurVBindGroup);
                pass3.draw(6);
                pass3.end();

                // Pass 4: Composite -> Screen
                const pass4 = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: context.getCurrentTexture().createView(),
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    }],
                });
                pass4.setPipeline(compositePipeline);
                pass4.setBindGroup(0, compositeBindGroup);
                pass4.draw(6);
                pass4.end();

                device.queue.submit([commandEncoder.finish()]);
                frameId = requestAnimationFrame(render);
            };

            frameId = requestAnimationFrame(render);

            return () => {
                cancelAnimationFrame(frameId);
                canvas.removeEventListener('mousedown', onMouseDown);
                window.removeEventListener('mouseup', onMouseUp);
                window.removeEventListener('mousemove', onMouseMove);
                canvas.removeEventListener('wheel', onWheel);
            }
        };

        initWebGPU();
    }, []);

    return (
        <>
            <canvas ref={canvasRef} />
            <div className="ui-controls">
                <h3>Atmosphere Settings</h3>
                {error ? (
                     <div style={{color: '#ef4444', padding: '10px', background: 'rgba(50,0,0,0.5)', borderRadius:'4px'}}>
                        {error}
                     </div>
                ) : (
                    <>
                        <div className="control-group">
                            <label>Sun Position <span>{sunAngle.toFixed(2)}</span></label>
                            <input 
                                type="range" 
                                min="0" max="1" step="0.001" 
                                value={sunAngle} 
                                onChange={(e) => setSunAngle(parseFloat(e.target.value))} 
                            />
                        </div>
                        
                        <div className="control-group">
                            <label>Scattering Intensity <span>{scatteringStrength.toFixed(1)}</span></label>
                            <input 
                                type="range" 
                                min="0" max="10" step="0.1" 
                                value={scatteringStrength} 
                                onChange={(e) => setScatteringStrength(parseFloat(e.target.value))} 
                            />
                        </div>

                        <div className="control-group">
                            <label>Bloom Intensity <span>{bloomStrength.toFixed(1)}</span></label>
                            <input 
                                type="range" 
                                min="0" max="3" step="0.1" 
                                value={bloomStrength} 
                                onChange={(e) => setBloomStrength(parseFloat(e.target.value))} 
                            />
                        </div>

                        <div className="control-group">
                            <label>Zoom <span>{zoom.toFixed(0)}</span></label>
                            <input 
                                type="range" 
                                min="1100" max="10000" step="10" 
                                value={zoom} 
                                onChange={(e) => setZoom(parseFloat(e.target.value))} 
                            />
                        </div>

                        <div className="hint">
                            Drag to rotate camera • Scroll to zoom
                        </div>
                    </>
                )}
            </div>
        </>
    );
};

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(<App />);