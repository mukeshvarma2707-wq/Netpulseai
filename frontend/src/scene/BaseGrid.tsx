import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { GRID_SIZE, cellToWorld } from "../lib/grid";
import { COLORS } from "../lib/theme";

const COUNT = GRID_SIZE * GRID_SIZE;

const vertexShader = /* glsl */ `
  attribute float aTint;
  attribute float aSeed;
  uniform float uTime;
  varying float vTint;
  varying float vAmb;
  varying vec2 vUv;
  void main() {
    vec4 wp = instanceMatrix * vec4(position, 1.0);
    // Very slow radial swell plus a rare, soft sparkle on ~1% of cells: the
    // grid feels alive when idle without competing with flagged towers.
    float wave = 0.5 + 0.5 * sin(uTime * 0.35 - length(wp.xz) * 0.09);
    float spark = step(0.99, aSeed) * pow(0.5 + 0.5 * sin(uTime * 0.5 + aSeed * 400.0), 8.0);
    vAmb = wave * 0.35 + spark;
    vTint = aTint;
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * wp;
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3 uTile;
  uniform vec3 uFlagged;
  uniform vec3 uSource;
  uniform float uSourceMix;
  varying float vTint;
  varying float vAmb;
  varying vec2 vUv;
  void main() {
    float edge = min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y));
    float inner = smoothstep(0.0, 0.07, edge);
    vec3 c = uTile * (1.0 + vAmb * 0.7);
    c = mix(c, uFlagged, step(0.5, vTint) * (1.0 - step(1.5, vTint)) * 0.55);
    c = mix(c, uSource, step(1.5, vTint) * uSourceMix);
    c *= mix(0.45, 1.0, inner);
    gl_FragColor = vec4(c, 1.0);
    #include <colorspace_fragment>
  }
`;

export type TileTint = Map<number, 1 | 2>; // 1 = has a case this hour, 2 = reallocation source

export function BaseGrid({ tints, sourceVisible }: { tints: TileTint; sourceVisible: boolean }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const geometry = useMemo(() => {
    const g = new THREE.PlaneGeometry(0.94, 0.94).rotateX(-Math.PI / 2);
    const seeds = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) seeds[i] = fract(Math.sin(i * 12.9898) * 43758.5453);
    g.setAttribute("aSeed", new THREE.InstancedBufferAttribute(seeds, 1));
    g.setAttribute("aTint", new THREE.InstancedBufferAttribute(new Float32Array(COUNT), 1));
    return g;
  }, []);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: {
          uTime: { value: 0 },
          uTile: { value: new THREE.Color(COLORS.tile) },
          uFlagged: { value: new THREE.Color("#1e2c44") },
          uSource: { value: new THREE.Color(COLORS.tileSource) },
          uSourceMix: { value: 0 },
        },
      }),
    [],
  );

  // Instance i is cell i + 1 at its real grid position.
  useEffect(() => {
    const mesh = meshRef.current!;
    const m = new THREE.Matrix4();
    for (let i = 0; i < COUNT; i++) {
      const [wx, wz] = cellToWorld(i + 1);
      mesh.setMatrixAt(i, m.makeTranslation(wx, 0, wz));
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, []);

  useEffect(() => {
    const attr = geometry.getAttribute("aTint") as THREE.InstancedBufferAttribute;
    (attr.array as Float32Array).fill(0);
    tints.forEach((t, cell) => {
      if (cell >= 1 && cell <= COUNT) attr.setX(cell - 1, t);
    });
    attr.needsUpdate = true;
  }, [tints, geometry]);

  useFrame(({ clock }, dt) => {
    material.uniforms.uTime.value = clock.elapsedTime;
    const goal = sourceVisible ? 0.75 : 0;
    material.uniforms.uSourceMix.value = THREE.MathUtils.damp(material.uniforms.uSourceMix.value, goal, 4, dt);
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, COUNT]}
      raycast={() => null}
      frustumCulled={false}
    />
  );
}

function fract(n: number) {
  return n - Math.floor(n);
}
