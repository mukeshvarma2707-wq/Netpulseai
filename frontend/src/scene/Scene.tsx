import { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html, Line } from "@react-three/drei";
import { Bloom, EffectComposer, Vignette } from "@react-three/postprocessing";
import * as THREE from "three";
import { BaseGrid, type TileTint } from "./BaseGrid";
import { Towers, type Tower } from "./Towers";
import { Beams, type Beam } from "./Beams";
import { CameraRig, HOME_VIEW, type CameraGoal } from "./CameraRig";
import { GRID_SIZE, cellToWorld } from "../lib/grid";
import { COLORS, STATUS_COLOR } from "../lib/theme";

interface Props {
  towers: Tower[];
  beams: Beam[];
  beamsVisible: boolean;
  tints: TileTint;
  selected: Tower | null;
  hovered: Tower | null;
  goal: CameraGoal;
  onHover: (id: number | null) => void;
  onSelect: (id: number | null) => void;
}

export function Scene({ towers, beams, beamsVisible, tints, selected, hovered, goal, onHover, onSelect }: Props) {
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [0, 170, 170], fov: 40, near: 0.5, far: 1000 }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      onPointerMissed={() => onSelect(null)}
      style={{ cursor: hovered ? "pointer" : "grab" }}
    >
      <color attach="background" args={[COLORS.bg]} />
      <CameraRig goal={goal} />
      <GridFrame />
      <BaseGrid tints={tints} sourceVisible={beamsVisible} />
      <Towers
        towers={towers}
        selectedId={selected?.id ?? null}
        hoveredId={hovered?.id ?? null}
        onHover={onHover}
        onSelect={onSelect}
      />
      <Beams beams={beams} selectedCaseId={selected?.id ?? null} visible={beamsVisible} />
      {selected && <SelectionMarker tower={selected} />}
      {hovered && hovered.id !== selected?.id && <TowerLabel tower={hovered} dim />}
      <EffectComposer multisampling={0}>
        <Bloom mipmapBlur luminanceThreshold={0.35} luminanceSmoothing={0.3} intensity={1.1} radius={0.7} />
        <Vignette offset={0.25} darkness={0.7} />
      </EffectComposer>
    </Canvas>
  );
}

export { HOME_VIEW };

/** Outline of the 100x100 grid plus a compass mark, so orientation is always clear. */
function GridFrame() {
  const h = GRID_SIZE / 2 + 0.6;
  const pts: [number, number, number][] = [
    [-h, 0, -h],
    [h, 0, -h],
    [h, 0, h],
    [-h, 0, h],
    [-h, 0, -h],
  ];
  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.05} raycast={() => null}>
        <planeGeometry args={[GRID_SIZE + 6, GRID_SIZE + 6]} />
        <meshBasicMaterial color="#070e1a" />
      </mesh>
      <Line points={pts} color="#1f3a5f" lineWidth={1.2} />
      <Html position={[0, 0, -h - 3]} center style={{ pointerEvents: "none" }}>
        <div className="font-mono text-[11px] tracking-[0.3em] text-sky-300/60">N</div>
      </Html>
    </group>
  );
}

function SelectionMarker({ tower }: { tower: Tower }) {
  const ring = useRef<THREE.Mesh>(null);
  const [wx, wz] = cellToWorld(tower.cellId);
  useFrame(({ clock }) => {
    if (!ring.current) return;
    const t = (clock.elapsedTime * 0.8) % 1;
    ring.current.scale.setScalar(0.6 + t * 2.2);
    (ring.current.material as THREE.MeshBasicMaterial).opacity = 0.9 * (1 - t);
  });
  return (
    <group position={[wx, 0.03, wz]}>
      <mesh ref={ring} rotation-x={-Math.PI / 2} raycast={() => null}>
        <ringGeometry args={[0.55, 0.68, 48]} />
        <meshBasicMaterial color={STATUS_COLOR[tower.status]} transparent toneMapped={false} />
      </mesh>
      <TowerLabel tower={tower} />
    </group>
  );
}

function TowerLabel({ tower, dim = false }: { tower: Tower; dim?: boolean }) {
  const [wx, wz] = cellToWorld(tower.cellId);
  const color = STATUS_COLOR[tower.status];
  // Positioned in world space when used on its own (hover); inside
  // SelectionMarker the parent group already carries the offset.
  const pos: [number, number, number] = dim ? [wx, tower.height + 0.9, wz] : [0, tower.height + 0.9, 0];
  return (
    <Html position={pos} center style={{ pointerEvents: "none" }} zIndexRange={[20, 0]}>
      <div
        className="whitespace-nowrap rounded border bg-[#07101d]/90 px-2 py-0.5 font-mono text-[11px] text-slate-200 shadow-lg backdrop-blur"
        style={{ borderColor: `${color}80`, opacity: dim ? 0.85 : 1 }}
      >
        <span style={{ color }}>●</span> CELL {tower.cellId}
      </div>
    </Html>
  );
}
