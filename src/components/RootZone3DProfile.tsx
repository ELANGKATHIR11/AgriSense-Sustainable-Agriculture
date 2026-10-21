import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Droplets, Eye, EyeOff, Layers } from 'lucide-react';
import { IrrigationZoneState } from '../api/client';
import { formatNumber } from '../utils/format';

// Pre-allocated module-scope objects (Zero per-frame GC allocation)
const _dummy = new THREE.Object3D();
const _layerColor = new THREE.Color();
const _wetSoilColor = new THREE.Color(0x285d78);

interface RootZone3DProfileProps {
  zone: IrrigationZoneState;
  nitrogenMgKg: number;
  phosphorusMgKg: number;
  potassiumMgKg: number;
}

export const RootZone3DProfile: React.FC<RootZone3DProfileProps> = React.memo(
  ({ zone, nitrogenMgKg, phosphorusMgKg, potassiumMgKg }) => {
    const { t, i18n } = useTranslation(['common']);
    const lang = i18n.language;

    const mountRef = useRef<HTMLDivElement | null>(null);
    const [webglReady, setWebglReady] = useState(true);
    const [exploded, setExploded] = useState(false);
    const [controlsCollapsed, setControlsCollapsed] = useState(false);

    const propsRef = useRef({ zone, exploded });
    useEffect(() => {
      propsRef.current = { zone, exploded };
    }, [zone, exploded]);

    useEffect(() => {
      const container = mountRef.current;
      if (!container) return;

      let renderer: THREE.WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({
          antialias: true,
          powerPreference: 'high-performance',
          stencil: false,
        });
      } catch {
        setWebglReady(false);
        return;
      }

      const isHeadless = navigator.webdriver || /HeadlessChrome/i.test(navigator.userAgent);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.12;
      renderer.setPixelRatio(isHeadless ? 1 : Math.min(window.devicePixelRatio, 1.5));

      const width = container.clientWidth || 460;
      const height = container.clientHeight || 320;
      renderer.setSize(width, height);
      container.innerHTML = '';
      container.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0d1912);
      scene.fog = new THREE.FogExp2(0x0d1912, 0.045);

      const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 50);
      camera.position.set(4.6, 2.4, 5.4);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(0, -0.3, 0);

      const ambient = new THREE.AmbientLight(0xdce8d5, 0.45);
      scene.add(ambient);
      const hemi = new THREE.HemisphereLight(0xdce8d5, 0x16291e, 0.95);
      scene.add(hemi);
      const dirLight = new THREE.DirectionalLight(0xfff3db, 2.1);
      dirLight.position.set(6, 9, 7);
      scene.add(dirLight);
      const fillLight = new THREE.DirectionalLight(0x55a3d1, 0.7);
      fillLight.position.set(-6, 4, -5);
      scene.add(fillLight);

      // 4 Soil Horizon Strata Block Meshes
      const strataDefs = [
        { name: 'O/A Horizon (0–15cm)', h: 0.45, baseY: 0.65, color: 0x3b2f26 },
        { name: 'Root Zone B1 (15–45cm)', h: 0.85, baseY: 0.0, color: 0x4d3a2c },
        { name: 'Subsoil Clay-Loam (45–80cm)', h: 0.75, baseY: -0.8, color: 0x5c4433 },
        { name: 'Capillary Aquifer (>80cm)', h: 0.45, baseY: -1.4, color: 0x245575 },
      ];

      const strataMeshes: { mesh: THREE.Mesh; baseY: number; mat: THREE.MeshStandardMaterial }[] =
        [];
      strataDefs.forEach((s) => {
        const geo = new THREE.BoxGeometry(2.8, s.h, 2.2);
        const mat = new THREE.MeshStandardMaterial({
          color: s.color,
          roughness: 0.82,
          metalness: 0.08,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.y = s.baseY;
        scene.add(mesh);
        strataMeshes.push({ mesh, baseY: s.baseY, mat });
      });

      // Composite Plant Canopy + Root Architecture Group
      const plantGroup = new THREE.Group();
      plantGroup.position.set(0, 0.88, 0);

      const stemMat = new THREE.MeshStandardMaterial({ color: 0x4a8c55, roughness: 0.55 });
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 1.35, 10), stemMat);
      stem.position.y = 0.65;
      plantGroup.add(stem);

      const leafGeo = new THREE.ConeGeometry(0.32, 0.75, 5);
      for (let i = 0; i < 5; i++) {
        const leaf = new THREE.Mesh(leafGeo, stemMat);
        leaf.position.set(
          Math.cos((i * Math.PI * 2) / 5) * 0.28,
          0.75 + (i % 2) * 0.25,
          Math.sin((i * Math.PI * 2) / 5) * 0.28
        );
        leaf.rotation.z = (i % 2 === 0 ? 1 : -1) * 0.55;
        plantGroup.add(leaf);
      }

      const rootMat = new THREE.MeshStandardMaterial({ color: 0xd6c099, roughness: 0.75 });
      const rootGeo = new THREE.CylinderGeometry(0.04, 0.01, 1.15, 6);
      for (let i = 0; i < 6; i++) {
        const angle = (i / 6) * Math.PI * 2;
        const rootStrand = new THREE.Mesh(rootGeo, rootMat);
        rootStrand.position.set(Math.cos(angle) * 0.22, -0.52, Math.sin(angle) * 0.22);
        rootStrand.rotation.z = Math.cos(angle) * 0.28;
        rootStrand.rotation.x = Math.sin(angle) * 0.28;
        plantGroup.add(rootStrand);
      }
      scene.add(plantGroup);

      // Surface Drip Lateral Pipe + Dielectric Probe
      const dripPipe = new THREE.Mesh(
        new THREE.CylinderGeometry(0.06, 0.06, 2.8, 10),
        new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.35, metalness: 0.6 })
      );
      dripPipe.rotation.z = Math.PI / 2;
      dripPipe.position.set(0, 0.94, 0.45);
      scene.add(dripPipe);

      const probeHead = new THREE.Mesh(
        new THREE.BoxGeometry(0.28, 0.22, 0.22),
        new THREE.MeshStandardMaterial({ color: 0x3b82a6, roughness: 0.3, metalness: 0.5 })
      );
      probeHead.position.set(-0.85, 1.0, -0.35);
      scene.add(probeHead);

      const probeRod = new THREE.Mesh(
        new THREE.CylinderGeometry(0.025, 0.025, 1.3, 8),
        new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.2, metalness: 0.85 })
      );
      probeRod.position.set(-0.85, 0.25, -0.35);
      scene.add(probeRod);

      // Infiltration Water Droplets InstancedMesh
      const dropCount = 32;
      const dropGeo = new THREE.SphereGeometry(0.055, 6, 6);
      const dropMat = new THREE.MeshBasicMaterial({ color: 0x55a3d1 });
      const dropInstanced = new THREE.InstancedMesh(dropGeo, dropMat, dropCount);
      dropInstanced.frustumCulled = false;
      scene.add(dropInstanced);

      const handleResize = () => {
        if (!container) return;
        const w = container.clientWidth || 460;
        const h = container.clientHeight || 320;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };

      const resizeObs = new ResizeObserver(() => {
        requestAnimationFrame(handleResize);
      });
      resizeObs.observe(container);

      let isVisible = true;
      const intersectObs = new IntersectionObserver(
        (entries) => {
          if (entries[0]) isVisible = entries[0].isIntersecting;
        },
        { threshold: 0.05 }
      );
      intersectObs.observe(container);

      const clock = new THREE.Clock();
      let frameId = 0;
      let currentExplode = 0;

      const animate = () => {
        frameId = requestAnimationFrame(animate);
        if (!isVisible || document.hidden) return;

        const delta = Math.min(clock.getDelta(), 0.1);
        const elapsed = clock.getElapsedTime();
        const { zone: zState, exploded: isExp } = propsRef.current;

        currentExplode = THREE.MathUtils.damp(currentExplode, isExp ? 1 : 0, 5, delta);

        for (let i = 0; i < strataMeshes.length; i++) {
          const s = strataMeshes[i];
          const offset = (1.5 - i) * 0.28 * currentExplode;
          s.mesh.position.y = s.baseY + offset;
        }

        // Tint root zone B1 stratum by live soilMoisturePct (Zero allocations)
        const moistRatio = THREE.MathUtils.clamp((zState.soilMoisturePct - 25) / 40, 0, 1);
        _layerColor.setHex(0x4d3a2c).lerp(_wetSoilColor, moistRatio * 0.65);
        strataMeshes[1].mat.color.lerp(_layerColor, 0.12);

        // Animate infiltrating water droplets
        const isActive = zState.valveStatus === 'active';
        const speed = isActive ? 0.9 : 0.25;
        const dropScale = isActive ? 1.15 : 0.65;
        for (let i = 0; i < dropCount; i++) {
          const prog = ((elapsed * speed + i / dropCount) % 1 + 1) % 1;
          const angle = (i * 2.4) % (Math.PI * 2);
          const rad = 0.15 + prog * 0.65;
          const dx = Math.cos(angle) * rad;
          const dz = 0.45 + Math.sin(angle) * (rad * 0.5);
          const dy = 0.88 - prog * 1.9;
          _dummy.position.set(dx, dy, dz);
          _dummy.scale.setScalar(dropScale);
          _dummy.updateMatrix();
          dropInstanced.setMatrixAt(i, _dummy.matrix);
        }
        dropInstanced.instanceMatrix.needsUpdate = true;

        controls.update();
        renderer.render(scene, camera);
      };
      animate();

      return () => {
        cancelAnimationFrame(frameId);
        resizeObs.disconnect();
        intersectObs.disconnect();
        controls.dispose();
        leafGeo.dispose();
        rootGeo.dispose();
        dropGeo.dispose();
        dropMat.dispose();
        renderer.dispose();
      };
    }, []);

    return (
      <div className="relative w-full h-[330px] rounded-xl overflow-hidden border border-[var(--agri-border)] bg-[#0D1912] text-white select-none">
        {/* Top-Left Telemetry Readout */}
        <div className="absolute top-3 left-3 z-20 px-3 py-1.5 rounded-lg bg-[#0B140E]/95 border border-white/15 text-xs font-mono-tabular flex items-center gap-2 pointer-events-none">
          <Droplets className="w-3.5 h-3.5 text-[#55A3D1]" />
          <span className="font-semibold">{zone.zoneId} Root-Zone Hydrology</span>
          <span className="text-white/40">·</span>
          <span className="text-[#5CA668]">{formatNumber(zone.soilMoisturePct, lang)}% VWC</span>
          <span className="text-white/40">·</span>
          <span>
            NPK {nitrogenMgKg}-{phosphorusMgKg}-{potassiumMgKg}
          </span>
        </div>

        {webglReady ? (
          <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
        ) : (
          <div className="w-full h-full flex items-center justify-center p-6 text-xs text-[#DCE8D5] font-mono-tabular">
            Root-Zone Profile: {zone.zoneName} · Moisture {zone.soilMoisturePct}% / Target{' '}
            {zone.targetMoisturePct}%
          </div>
        )}

        {/* Collapsible Bottom Control Dock */}
        <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center">
          <button
            type="button"
            onClick={() => setControlsCollapsed((c) => !c)}
            className="px-2.5 py-0.5 rounded-t bg-[#0B140E] border border-b-0 border-white/15 text-[10px] font-mono-tabular text-[#DCE8D5] cursor-pointer flex items-center gap-1"
          >
            {controlsCollapsed ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            <span>
              {controlsCollapsed
                ? t('common:hud3d.showControls')
                : t('common:hud3d.hideControls')}
            </span>
          </button>

          {!controlsCollapsed && (
            <div className="rounded-lg bg-[#0B140E]/95 border border-white/15 px-3 py-1.5 flex items-center gap-2 text-[11px]">
              <button
                type="button"
                onClick={() => setExploded((e) => !e)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded font-medium cursor-pointer ${
                  exploded
                    ? 'bg-[#4A8C55] text-white'
                    : 'bg-white/10 text-[#DCE8D5] hover:text-white'
                }`}
              >
                <Layers className="w-3 h-3" />
                <span>{exploded ? 'Compact Horizons' : 'Explode Soil Horizons'}</span>
              </button>
              <span className="text-white/40">|</span>
              <span className="font-mono-tabular text-[#DCE8D5]/80">
                Flow: {formatNumber(zone.flowRateLpm, lang)} L/min
              </span>
            </div>
          )}
        </div>
      </div>
    );
  }
);
