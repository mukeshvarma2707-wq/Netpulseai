import { useEffect, useMemo, useRef } from "react";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { cellToWorld, type TowerStatus } from "../lib/grid";
import { COLORS, STATUS_COLOR } from "../lib/theme";

export interface Tower {
  id: number;
  cellId: number;
  status: TowerStatus;
  height: number;
  visible: boolean;
}

const vertexShader = /* glsl */ `
  attribute vec3 aColor;
  attribute float aPhase;
  attribute float aPulse;
  attribute float aFade;
  attribute float aHi;
  uniform float uTime;
  varying vec3 vColor;
  varying vec3 vNormal2;
  varying float vH;
  varying float vFade;
  varying float vPulse;
  varying float vHi;
  void main() {
    vH = position.y;
    vColor = aColor;
    vFade = aFade;
    vHi = aHi;
    vNormal2 = normal;
    vPulse = aPulse > 0.0 ? 0.5 + 0.5 * sin(uTime * aPulse + aPhase) : 0.0;
    gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying vec3 vNormal2;
  varying float vH;
  varying float vFade;
  varying float vPulse;
  varying float vHi;
  void main() {
    float shade = abs(vNormal2.y) > 0.5 ? (vNormal2.y > 0.0 ? 1.35 : 0.4) : 0.72 + 0.28 * abs(vNormal2.x);
    float grad = 0.3 + 0.9 * pow(vH, 1.3);
    float glow = 0.75 + 0.5 * vPulse + 0.9 * vHi;
    vec3 c = vColor * grad * shade * glow;
    c += vColor * smoothstep(0.88, 1.0, vH) * (0.7 + vHi);
    gl_FragColor = vec4(c, vFade);
    #include <colorspace_fragment>
  }
`;

const PULSE_SPEED: Record<TowerStatus, number> = { routine: 1.6, anomalous: 3.4, resolved: 0 };
const _m = new THREE.Matrix4();
const _c = new THREE.Color();

interface Props {
  towers: Tower[];
  selectedId: number | null;
  hoveredId: number | null;
  onHover: (id: number | null) => void;
  onSelect: (id: number) => void;
}

export function Towers({ towers, selectedId, hoveredId, onHover, onSelect }: Props) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = Math.max(1, towers.length);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: { uTime: { value: 0 } },
        transparent: true,
      }),
    [],
  );

  // Per-instance buffers, rebuilt whenever the set of towers changes (new hour).
  const { geometry, anim } = useMemo(() => {
    const g = new THREE.BoxGeometry(0.62, 1, 0.62).translate(0, 0.5, 0);
    const colors = new Float32Array(count * 3);
    const phase = new Float32Array(count);
    const pulse = new Float32Array(count);
    const target = new Float32Array(count * 3);
    towers.forEach((t, i) => {
      // Resolved towers start amber and settle to green as the fix "arrives".
      _c.set(t.status === "resolved" ? COLORS.routine : STATUS_COLOR[t.status]).toArray(colors, i * 3);
      _c.set(STATUS_COLOR[t.status]).toArray(target, i * 3);
      phase[i] = (t.cellId * 0.618) % (Math.PI * 2);
      pulse[i] = PULSE_SPEED[t.status];
    });
    g.setAttribute("aColor", new THREE.InstancedBufferAttribute(colors, 3));
    g.setAttribute("aPhase", new THREE.InstancedBufferAttribute(phase, 1));
    g.setAttribute("aPulse", new THREE.InstancedBufferAttribute(pulse, 1));
    g.setAttribute("aFade", new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
    g.setAttribute("aHi", new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
    return { geometry: g, anim: { height: new Float32Array(count), targetColor: target, settled: false } };
  }, [towers.map((t) => `${t.id}:${t.status}`).join("|")]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  // Anything that changes a target (filter, selection) restarts the animation loop.
  useEffect(() => {
    anim.settled = false;
  }, [towers, selectedId, hoveredId, anim]);

  useFrame(({ clock }, dt) => {
    material.uniforms.uTime.value = clock.elapsedTime;
    const mesh = meshRef.current;
    if (!mesh || anim.settled) return;

    const fade = geometry.getAttribute("aFade") as THREE.InstancedBufferAttribute;
    const hi = geometry.getAttribute("aHi") as THREE.InstancedBufferAttribute;
    const col = geometry.getAttribute("aColor") as THREE.InstancedBufferAttribute;
    const fa = fade.array as Float32Array;
    const ha = hi.array as Float32Array;
    const ca = col.array as Float32Array;
    let moving = false;

    towers.forEach((t, i) => {
      const fGoal = t.visible ? 1 : 0.07;
      const hGoal = t.id === selectedId || t.id === hoveredId ? 1 : 0;
      fa[i] = step(fa[i], fGoal, 5, dt);
      ha[i] = step(ha[i], hGoal, 8, dt);
      anim.height[i] = step(anim.height[i], t.height, 3.2, dt);
      for (let k = 0; k < 3; k++) ca[i * 3 + k] = step(ca[i * 3 + k], anim.targetColor[i * 3 + k], 0.9, dt);
      moving ||=
        Math.abs(fa[i] - fGoal) > 1e-3 ||
        Math.abs(ha[i] - hGoal) > 1e-3 ||
        Math.abs(anim.height[i] - t.height) > 1e-3 ||
        Math.abs(ca[i * 3] - anim.targetColor[i * 3]) > 1e-3;

      const [wx, wz] = cellToWorld(t.cellId);
      const sy = Math.max(0.001, anim.height[i] * (0.15 + 0.85 * fa[i]));
      mesh.setMatrixAt(i, _m.makeScale(1, sy, 1).setPosition(wx, 0, wz));
    });

    mesh.instanceMatrix.needsUpdate = true;
    fade.needsUpdate = hi.needsUpdate = col.needsUpdate = true;
    mesh.computeBoundingSphere();
    anim.settled = !moving;
  });

  // Raycast hits against faded-out towers are ignored.
  const pick = (e: ThreeEvent<PointerEvent | MouseEvent>) =>
    e.intersections.find((h) => h.object === meshRef.current && h.instanceId != null && towers[h.instanceId]?.visible);

  return (
    <instancedMesh
      key={count}
      ref={meshRef}
      args={[geometry, material, count]}
      count={towers.length}
      frustumCulled={false}
      onPointerMove={(e) => {
        const hit = pick(e);
        if (!hit) return onHover(null);
        e.stopPropagation();
        onHover(towers[hit.instanceId!].id);
      }}
      onPointerOut={() => onHover(null)}
      onClick={(e) => {
        const hit = pick(e);
        if (!hit) return;
        e.stopPropagation();
        onSelect(towers[hit.instanceId!].id);
      }}
    />
  );
}

function step(current: number, goal: number, lambda: number, dt: number) {
  const next = THREE.MathUtils.damp(current, goal, lambda, dt);
  return Math.abs(next - goal) < 1e-4 ? goal : next;
}
