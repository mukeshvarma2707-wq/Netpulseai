import { useEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { cellToWorld } from "../lib/grid";
import { COLORS } from "../lib/theme";

export interface Beam {
  caseId: number;
  from: number; // source cell id
  to: number; // congested cell id
  amount: number;
}

// Quadratic arc from source tile up and over onto the congested tower's base.
const arc = /* glsl */ `
  vec3 arcPoint(vec3 a, vec3 b, float t) {
    vec3 mid = mix(a, b, 0.5);
    mid.y += 0.7 + 0.45 * distance(a.xz, b.xz);
    return mix(mix(a, mid, t), mix(mid, b, t), t);
  }
`;

const particleVertex = /* glsl */ `
  attribute vec3 aStart;
  attribute vec3 aEnd;
  attribute float aOffset;
  attribute float aSpeed;
  attribute float aSize;
  attribute float aSel;
  uniform float uTime;
  uniform float uPixelRatio;
  varying float vT;
  varying float vSel;
  ${arc}
  void main() {
    float t = fract(aOffset + uTime * aSpeed);
    vT = t;
    vSel = aSel;
    vec4 mv = modelViewMatrix * vec4(arcPoint(aStart, aEnd, t), 1.0);
    gl_PointSize = max(1.5, aSize * (1.0 + aSel * 0.8) * (90.0 / -mv.z)) * uPixelRatio;
    gl_Position = projectionMatrix * mv;
  }
`;

const particleFragment = /* glsl */ `
  uniform vec3 uFrom;
  uniform vec3 uTo;
  uniform float uOpacity;
  uniform float uDim;
  varying float vT;
  varying float vSel;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.0, d) * sin(vT * 3.14159);
    a *= uOpacity * mix(uDim, 1.0, vSel);
    vec3 c = mix(uFrom, uTo, vT) * (1.3 + vSel);
    gl_FragColor = vec4(c, a);
    #include <colorspace_fragment>
  }
`;

const SEGMENTS = 14;

interface Props {
  beams: Beam[];
  selectedCaseId: number | null;
  visible: boolean;
}

export function Beams({ beams, selectedCaseId, visible }: Props) {
  const dpr = useThree((s) => s.viewport.dpr);
  const hasSelection = beams.some((b) => b.caseId === selectedCaseId);

  const particleMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: particleVertex,
        fragmentShader: particleFragment,
        uniforms: {
          uTime: { value: 0 },
          uPixelRatio: { value: 1 },
          uFrom: { value: new THREE.Color(COLORS.beam) },
          uTo: { value: new THREE.Color(COLORS.resolved) },
          uOpacity: { value: 0 },
          uDim: { value: 1 },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  const lineMat = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        toneMapped: false,
      }),
    [],
  );

  const { points, lines } = useMemo(() => {
    const start: number[] = [];
    const end: number[] = [];
    const offset: number[] = [];
    const speed: number[] = [];
    const size: number[] = [];
    const sel: number[] = [];
    const linePos: number[] = [];
    const lineCol: number[] = [];
    const cyan = new THREE.Color(COLORS.beam);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const p = new THREE.Vector3();

    for (const beam of beams) {
      const [ax, az] = cellToWorld(beam.from);
      const [bx, bz] = cellToWorld(beam.to);
      a.set(ax, 0.05, az);
      b.set(bx, 0.25, bz);
      const isSel = beam.caseId === selectedCaseId ? 1 : 0;

      // Particle density and size both scale with the capacity moved.
      const n = Math.min(40, Math.max(3, Math.round(3 + 2.2 * Math.sqrt(beam.amount))));
      const s = 1.6 + 1.1 * Math.log10(1 + beam.amount);
      for (let i = 0; i < n; i++) {
        start.push(a.x, a.y, a.z);
        end.push(b.x, b.y, b.z);
        offset.push(i / n + Math.random() * (0.5 / n));
        speed.push(0.32 + Math.random() * 0.06);
        size.push(s);
        sel.push(isSel);
      }

      const intensity = isSel ? 1 : 0.35;
      for (let i = 0; i < SEGMENTS; i++) {
        for (const t of [i / SEGMENTS, (i + 1) / SEGMENTS]) {
          arcPoint(a, b, t, p);
          linePos.push(p.x, p.y, p.z);
          lineCol.push(cyan.r * intensity, cyan.g * intensity, cyan.b * intensity);
        }
      }
    }

    const pg = new THREE.BufferGeometry();
    pg.setAttribute("position", new THREE.Float32BufferAttribute(start, 3));
    pg.setAttribute("aStart", new THREE.Float32BufferAttribute(start, 3));
    pg.setAttribute("aEnd", new THREE.Float32BufferAttribute(end, 3));
    pg.setAttribute("aOffset", new THREE.Float32BufferAttribute(offset, 1));
    pg.setAttribute("aSpeed", new THREE.Float32BufferAttribute(speed, 1));
    pg.setAttribute("aSize", new THREE.Float32BufferAttribute(size, 1));
    pg.setAttribute("aSel", new THREE.Float32BufferAttribute(sel, 1));

    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.Float32BufferAttribute(linePos, 3));
    lg.setAttribute("color", new THREE.Float32BufferAttribute(lineCol, 3));
    return { points: pg, lines: lg };
  }, [beams, selectedCaseId]);

  useEffect(
    () => () => {
      points.dispose();
      lines.dispose();
    },
    [points, lines],
  );

  useFrame(({ clock }, dt) => {
    const u = particleMat.uniforms;
    u.uTime.value = clock.elapsedTime;
    u.uPixelRatio.value = dpr;
    u.uOpacity.value = THREE.MathUtils.damp(u.uOpacity.value, visible ? 1 : 0, 4, dt);
    // With a case selected, its beams stay bright and the rest recede.
    u.uDim.value = THREE.MathUtils.damp(u.uDim.value, hasSelection ? 0.25 : 1, 4, dt);
    lineMat.opacity = u.uOpacity.value * (hasSelection ? 0.9 : 0.6);
  });

  return (
    <group>
      <lineSegments geometry={lines} material={lineMat} raycast={() => null} frustumCulled={false} />
      <points geometry={points} material={particleMat} raycast={() => null} frustumCulled={false} />
    </group>
  );
}

function arcPoint(a: THREE.Vector3, b: THREE.Vector3, t: number, out: THREE.Vector3) {
  const mid = a.clone().lerp(b, 0.5);
  mid.y += 0.7 + 0.45 * Math.hypot(a.x - b.x, a.z - b.z);
  const p0 = a.clone().lerp(mid, t);
  const p1 = mid.clone().lerp(b, t);
  return out.copy(p0.lerp(p1, t));
}
