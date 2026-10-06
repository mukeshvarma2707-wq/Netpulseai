import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

export interface CameraGoal {
  position: [number, number, number];
  target: [number, number, number];
  nonce: number; // bump to re-trigger a flight to the same spot
}

export const HOME_VIEW = { position: [0, 88, 82] as [number, number, number], target: [0, 0, 4] as [number, number, number] };

/** Orbit controls plus a damped "fly to" that the user can interrupt by dragging. */
export function CameraRig({ goal }: { goal: CameraGoal }) {
  const controls = useRef<OrbitControlsImpl>(null);
  const camera = useThree((s) => s.camera);
  const flight = useRef<{ pos: THREE.Vector3; target: THREE.Vector3 } | null>(null);

  useEffect(() => {
    flight.current = { pos: new THREE.Vector3(...goal.position), target: new THREE.Vector3(...goal.target) };
  }, [goal]);

  useFrame((_, dt) => {
    const f = flight.current;
    const c = controls.current;
    if (!f || !c) return;
    const k = 1 - Math.exp(-dt * 2.6);
    camera.position.lerp(f.pos, k);
    c.target.lerp(f.target, k);
    c.update();
    if (camera.position.distanceTo(f.pos) < 0.05 && c.target.distanceTo(f.target) < 0.05) flight.current = null;
  });

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={6}
      maxDistance={190}
      maxPolarAngle={Math.PI * 0.44}
      screenSpacePanning={false}
      onStart={() => (flight.current = null)}
    />
  );
}
