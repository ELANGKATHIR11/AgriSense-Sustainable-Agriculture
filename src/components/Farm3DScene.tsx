import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  Box,
  Compass,
  Droplets,
  Eye,
  EyeOff,
  Layers,
  Map as MapIcon,
  Radio,
  Sprout,
} from 'lucide-react';
import { DeviceTelemetry, DigitalTwinSeriesPoint, IrrigationZoneState } from '../api/client';
import { formatNumber } from '../utils/format';

// Pre-allocated module-scope math helpers (Zero GC in animation loop)
const _camTargetPos = new THREE.Vector3();
const _camLookTarget = new THREE.Vector3();
const _colorDry = new THREE.Color(0x8c6239);
const _colorOptimal = new THREE.Color(0x2d6a42);
const _colorWet = new THREE.Color(0x2b729f);
const _tempColor = new THREE.Color();
const _dummyObj = new THREE.Object3D();

// Cached singleton procedural soil canvas so page navigation never re-runs 2D canvas drawing loops
let _sharedSoilCanvas: HTMLCanvasElement | null = null;
function getSharedSoilCanvas(): HTMLCanvasElement {
  if (_sharedSoilCanvas) return _sharedSoilCanvas;
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#243c2c';
    ctx.fillRect(0, 0, 256, 256);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 256; i += 16) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(256, i);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, 256);
      ctx.stroke();
    }
    for (let i = 0; i < 450; i++) {
      const x = (i * 73) % 256;
      const y = (i * 137) % 256;
      ctx.fillStyle = i % 2 === 0 ? 'rgba(18, 32, 22, 0.25)' : 'rgba(74, 140, 85, 0.15)';
      ctx.fillRect(x, y, 2, 2);
    }
  }
  _sharedSoilCanvas = canvas;
  return canvas;
}

export type CameraPresetKey =
  | 'overview'
  | 'ZONE-A'
  | 'ZONE-B'
  | 'ZONE-C'
  | 'ZONE-D'
  | 'ZONE-CENTRAL';

interface CameraPreset {
  pos: [number, number, number];
  lookAt: [number, number, number];
}

const CAMERA_PRESETS: Record<CameraPresetKey, CameraPreset> = {
  overview: { pos: [0, 13.5, 16.5], lookAt: [0, 0.2, 0] },
  'ZONE-A': { pos: [-9.2, 6.2, -0.5], lookAt: [-5.5, 0.4, -4.2] },
  'ZONE-B': { pos: [9.4, 6.4, 0.2], lookAt: [5.2, 0.6, -3.8] },
  'ZONE-C': { pos: [9.5, 6.5, 8.8], lookAt: [5.4, 0.8, 4.4] },
  'ZONE-D': { pos: [-9.2, 6.4, 8.6], lookAt: [-5.4, 0.6, 4.2] },
  'ZONE-CENTRAL': { pos: [0, 5.2, 6.8], lookAt: [0, 1.1, 0] },
};

const TOWER_NODES_SPEC = [
  { id: 'ESP32-NODE-101', zoneId: 'ZONE-A', name: 'Block A Paddy', x: -5.5, z: -4.2 },
  { id: 'ESP32-NODE-102', zoneId: 'ZONE-B', name: 'Block B Polyhouse', x: 5.2, z: -3.8 },
  { id: 'ESP32-NODE-103', zoneId: 'ZONE-C', name: 'Block C Millet', x: 5.4, z: 4.4 },
  { id: 'ESP32-NODE-104', zoneId: 'ZONE-D', name: 'Block D Turmeric', x: -5.4, z: 4.2 },
] as const;

export interface Farm3DSceneProps {
  devices: DeviceTelemetry[];
  zones: IrrigationZoneState[];
  tankLevelPct: number;
  selectedZoneId?: string;
  onSelectZone?: (zoneId: string) => void;
  onSelectDevice?: (device: DeviceTelemetry) => void;
  heightClass?: string;
  /** Optional 4D Digital Twin time-series for temporal scrubbing on Page 5 */
  twinSeries?: DigitalTwinSeriesPoint[];
  selectedDayIndex?: number;
  onChangeDayIndex?: (index: number) => void;
}

function getGroundHeight(x: number, z: number): number {
  const ridge = Math.sin(x * 0.22) * Math.cos(z * 0.22) * 0.35;
  const terraceSlope = (z + 8) * 0.03 + (x > 0 ? 0.18 : 0);
  const edgeRise =
    Math.max(0, (Math.abs(x) - 9.5) * 0.38) + Math.max(0, (Math.abs(z) - 8.5) * 0.42);
  return ridge + terraceSlope + edgeRise;
}

function createProceduralSoilTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(getSharedSoilCanvas());
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(6, 6);
  tex.anisotropy = Math.min(4, maxAnisotropy);
  return tex;
}

export const Farm3DScene: React.FC<Farm3DSceneProps> = React.memo(
  ({
    devices,
    zones,
    tankLevelPct,
    selectedZoneId,
    onSelectZone,
    onSelectDevice,
    heightClass = 'h-[430px]',
    twinSeries,
    selectedDayIndex = 0,
    onChangeDayIndex,
  }) => {
    const { t, i18n } = useTranslation(['common']);
    const lang = i18n.language;

    const canvasMountRef = useRef<HTMLDivElement | null>(null);
    const [webglSupported, setWebglSupported] = useState(true);
    const [viewMode, setViewMode] = useState<'3d' | '2d'>('3d');
    const [controlsCollapsed, setControlsCollapsed] = useState(false);
    const [cameraPreset, setCameraPreset] = useState<CameraPresetKey>('overview');

    // Interactive 3D Layer Toggles
    const [showMoistureHeatmap, setShowMoistureHeatmap] = useState(true);
    const [showIrrigationFlow, setShowIrrigationFlow] = useState(true);
    const [showCropCanopy, setShowCropCanopy] = useState(true);
    const [showSensorTowers, setShowSensorTowers] = useState(true);

    const [inspectedLabel, setInspectedLabel] = useState<string>(
      'ZONE-A · Block A Paddy (54.2% VWC)'
    );

    // Pre-indexed O(1) lookup tables so the 60fps animation loop does zero array.find() scans
    const stateRef = useRef({
      devices,
      zones,
      moistureByZone: {} as Record<string, number>,
      deviceById: {} as Record<string, DeviceTelemetry>,
      valveActiveByZone: {} as Record<string, boolean>,
      tankLevelPct,
      cameraPreset,
      showMoistureHeatmap,
      showIrrigationFlow,
      showCropCanopy,
      showSensorTowers,
      twinSeries,
      selectedDayIndex,
      onSelectZone,
      onSelectDevice,
    });

    useEffect(() => {
      const moistureByZone: Record<string, number> = {};
      const deviceById: Record<string, DeviceTelemetry> = {};
      for (let i = 0; i < devices.length; i++) {
        const d = devices[i];
        moistureByZone[d.zoneId] = d.readings.soilMoisturePct;
        deviceById[d.id] = d;
      }
      const valveActiveByZone: Record<string, boolean> = {};
      for (let i = 0; i < zones.length; i++) {
        const z = zones[i];
        valveActiveByZone[z.zoneId] = z.valveStatus === 'active';
      }

      stateRef.current = {
        devices,
        zones,
        moistureByZone,
        deviceById,
        valveActiveByZone,
        tankLevelPct,
        cameraPreset,
        showMoistureHeatmap,
        showIrrigationFlow,
        showCropCanopy,
        showSensorTowers,
        twinSeries,
        selectedDayIndex,
        onSelectZone,
        onSelectDevice,
      };
    }, [
      devices,
      zones,
      tankLevelPct,
      cameraPreset,
      showMoistureHeatmap,
      showIrrigationFlow,
      showCropCanopy,
      showSensorTowers,
      twinSeries,
      selectedDayIndex,
      onSelectZone,
      onSelectDevice,
    ]);

    useEffect(() => {
      if (selectedZoneId && CAMERA_PRESETS[selectedZoneId as CameraPresetKey]) {
        setCameraPreset(selectedZoneId as CameraPresetKey);
      }
    }, [selectedZoneId]);

    // Keyboard shortcut 'H' to toggle HUD dock collapse
    useEffect(() => {
      const onKeyDown = (e: KeyboardEvent) => {
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if (e.key === 'h' || e.key === 'H') {
          setControlsCollapsed((prev) => !prev);
        }
      };
      window.addEventListener('keydown', onKeyDown);
      return () => window.removeEventListener('keydown', onKeyDown);
    }, []);

    useEffect(() => {
      if (viewMode !== '3d') return;
      const mountEl = canvasMountRef.current;
      if (!mountEl) return;

      let renderer: THREE.WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({
          antialias: true,
          powerPreference: 'high-performance',
          stencil: false,
          depth: true,
        });
      } catch {
        setWebglSupported(false);
        setViewMode('2d');
        return;
      }

      const isHeadless =
        navigator.webdriver || /HeadlessChrome/i.test(navigator.userAgent);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.15;
      renderer.setPixelRatio(isHeadless ? 1 : Math.min(window.devicePixelRatio, 1.5));
      renderer.shadowMap.enabled = false;

      const width = mountEl.clientWidth || 800;
      const height = mountEl.clientHeight || 430;
      renderer.setSize(width, height);
      mountEl.innerHTML = '';
      mountEl.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0d1912);
      scene.fog = new THREE.FogExp2(0x0d1912, 0.021);

      const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 110);
      camera.position.set(0, 13.5, 16.5);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.maxPolarAngle = Math.PI / 2 - 0.05;
      controls.minDistance = 4.5;
      controls.maxDistance = 32;
      controls.target.set(0, 0.2, 0);

      // Fast 3-Point Warm/Cool Lighting Rig (No synchronous PMREM cubemap stalls)
      const ambientLight = new THREE.AmbientLight(0xdce8d5, 0.45);
      scene.add(ambientLight);

      const hemiLight = new THREE.HemisphereLight(0xdce8d5, 0x172b1e, 0.95);
      scene.add(hemiLight);

      const keyLight = new THREE.DirectionalLight(0xfff4e0, 2.3);
      keyLight.position.set(14, 22, 12);
      scene.add(keyLight);

      const rimLight = new THREE.DirectionalLight(0x55a3d1, 0.85);
      rimLight.position.set(-14, 10, -12);
      scene.add(rimLight);

      const maxAniso = renderer.capabilities.getMaxAnisotropy();
      const soilTexture = createProceduralSoilTexture(maxAniso);

      // 1. Contoured Outdoor Terrain Mesh
      const terrainGeo = new THREE.PlaneGeometry(34, 30, 44, 40);
      terrainGeo.rotateX(-Math.PI / 2);
      const posAttr = terrainGeo.attributes.position;
      const colorsArray = new Float32Array(posAttr.count * 3);

      for (let i = 0; i < posAttr.count; i++) {
        const vx = posAttr.getX(i);
        const vz = posAttr.getZ(i);
        const vy = getGroundHeight(vx, vz);
        posAttr.setY(i, vy);
        colorsArray[i * 3] = 0.18;
        colorsArray[i * 3 + 1] = 0.38;
        colorsArray[i * 3 + 2] = 0.24;
      }
      terrainGeo.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
      terrainGeo.computeVertexNormals();

      const terrainMat = new THREE.MeshStandardMaterial({
        map: soilTexture,
        vertexColors: true,
        roughness: 0.88,
        metalness: 0.05,
      });
      const terrainMesh = new THREE.Mesh(terrainGeo, terrainMat);
      scene.add(terrainMesh);

      // 2. Four Distinct Farm Plot Raised Beds + Raycast Proxies (Layer 1)
      const clickableObjects: THREE.Object3D[] = [];
      const plotDefs = [
        { id: 'ZONE-A', name: 'Block A Paddy', x: -5.5, z: -4.2, w: 7.4, d: 5.8, color: 0x2d6a42 },
        { id: 'ZONE-B', name: 'Block B Polyhouse', x: 5.2, z: -3.8, w: 7.2, d: 5.6, color: 0x3b7a57 },
        { id: 'ZONE-C', name: 'Block C Millet', x: 5.4, z: 4.4, w: 7.2, d: 5.6, color: 0x6c7c3c },
        { id: 'ZONE-D', name: 'Block D Turmeric', x: -5.4, z: 4.2, w: 7.4, d: 5.6, color: 0x4e7c44 },
      ];

      const plotMats: Record<string, THREE.MeshStandardMaterial> = {};
      const sharedBedGeo = new THREE.BoxGeometry(1, 0.22, 1);

      plotDefs.forEach((p) => {
        const py = getGroundHeight(p.x, p.z);
        const bedMat = new THREE.MeshStandardMaterial({
          color: p.color,
          roughness: 0.78,
          metalness: 0.08,
        });
        plotMats[p.id] = bedMat;
        const bedMesh = new THREE.Mesh(sharedBedGeo, bedMat);
        bedMesh.scale.set(p.w, 1, p.d);
        bedMesh.position.set(p.x, py + 0.06, p.z);
        bedMesh.layers.enable(1);
        bedMesh.userData = {
          type: 'zone',
          zoneId: p.id,
          label: `${p.id} · ${p.name}`,
        };
        scene.add(bedMesh);
        clickableObjects.push(bedMesh);
      });

      // 3. Composite Procedural Central Reservoir & Edge Gateway Station
      const reservoirGroup = new THREE.Group();
      const centerGroundY = getGroundHeight(0, 0);
      reservoirGroup.position.set(0, centerGroundY, 0);

      const plinth = new THREE.Mesh(
        new THREE.CylinderGeometry(1.65, 1.8, 0.28, 20),
        new THREE.MeshStandardMaterial({ color: 0x767d75, roughness: 0.85, metalness: 0.1 })
      );
      plinth.position.y = 0.14;
      reservoirGroup.add(plinth);

      const tankShell = new THREE.Mesh(
        new THREE.CylinderGeometry(1.35, 1.35, 2.1, 24),
        new THREE.MeshStandardMaterial({ color: 0x264e3a, roughness: 0.42, metalness: 0.58 })
      );
      tankShell.position.y = 1.33;
      reservoirGroup.add(tankShell);

      // Lightweight transparent StandardMaterial (avoids multi-pass transmission framebuffer lag)
      const waterColumnMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(1.38, 1.38, 1.6, 20, 1, true, 0, Math.PI * 0.35),
        new THREE.MeshStandardMaterial({
          color: 0x438eba,
          emissive: 0x174866,
          roughness: 0.2,
          metalness: 0.15,
          transparent: true,
          opacity: 0.88,
        })
      );
      waterColumnMesh.position.y = 1.1;
      reservoirGroup.add(waterColumnMesh);

      const tankRoof = new THREE.Mesh(
        new THREE.ConeGeometry(1.48, 0.55, 24),
        new THREE.MeshStandardMaterial({ color: 0x9aa59e, roughness: 0.35, metalness: 0.65 })
      );
      tankRoof.position.y = 2.65;
      reservoirGroup.add(tankRoof);

      const ventStack = new THREE.Mesh(
        new THREE.CylinderGeometry(0.12, 0.16, 0.45, 10),
        new THREE.MeshStandardMaterial({ color: 0xc8d0cb, roughness: 0.3, metalness: 0.7 })
      );
      ventStack.position.y = 3.02;
      reservoirGroup.add(ventStack);

      const gatewayBox = new THREE.Mesh(
        new THREE.BoxGeometry(0.55, 0.72, 0.38),
        new THREE.MeshStandardMaterial({ color: 0xdce8d5, roughness: 0.35, metalness: 0.3 })
      );
      gatewayBox.position.set(1.42, 0.95, 0.3);
      reservoirGroup.add(gatewayBox);

      const pumpBody = new THREE.Mesh(
        new THREE.CylinderGeometry(0.28, 0.28, 0.65, 12),
        new THREE.MeshStandardMaterial({ color: 0x3b82a6, roughness: 0.32, metalness: 0.6 })
      );
      pumpBody.rotation.z = Math.PI / 2;
      pumpBody.position.set(0, 0.45, 1.52);
      reservoirGroup.add(pumpBody);

      const gwMast = new THREE.Mesh(
        new THREE.CylinderGeometry(0.03, 0.04, 1.4, 8),
        new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.25, metalness: 0.75 })
      );
      gwMast.position.set(1.42, 1.9, 0.3);
      reservoirGroup.add(gwMast);

      tankShell.layers.enable(1);
      tankShell.userData = {
        type: 'zone',
        zoneId: 'ZONE-CENTRAL',
        label: 'RPI-GW-01 · Central Reservoir & Weather Station',
      };
      clickableObjects.push(tankShell);
      scene.add(reservoirGroup);

      // 4. Composite Block B Polyhouse Structure (Zero transmission pass overhead)
      const polyhouseGroup = new THREE.Group();
      const polyY = getGroundHeight(5.2, -3.8);
      polyhouseGroup.position.set(5.2, polyY + 0.16, -3.8);

      const archGeo = new THREE.TorusGeometry(2.4, 0.045, 6, 18, Math.PI);
      const archMat = new THREE.MeshStandardMaterial({
        color: 0xcbd5e1,
        roughness: 0.3,
        metalness: 0.75,
      });
      for (let zOff = -2.2; zOff <= 2.2; zOff += 1.1) {
        const arch = new THREE.Mesh(archGeo, archMat);
        arch.position.set(0, 0, zOff);
        polyhouseGroup.add(arch);
      }

      const ridgeBeam = new THREE.Mesh(
        new THREE.CylinderGeometry(0.04, 0.04, 4.6, 8),
        archMat
      );
      ridgeBeam.rotation.x = Math.PI / 2;
      ridgeBeam.position.set(0, 2.4, 0);
      polyhouseGroup.add(ridgeBeam);

      const vaultGeo = new THREE.CylinderGeometry(
        2.38,
        2.38,
        4.5,
        18,
        1,
        false,
        0,
        Math.PI
      );
      vaultGeo.rotateZ(-Math.PI / 2);
      vaultGeo.rotateY(Math.PI / 2);
      const vaultMat = new THREE.MeshStandardMaterial({
        color: 0xdff2e6,
        emissive: 0x163022,
        roughness: 0.25,
        metalness: 0.1,
        transparent: true,
        opacity: 0.34,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const vaultMesh = new THREE.Mesh(vaultGeo, vaultMat);
      vaultMesh.renderOrder = 2;
      polyhouseGroup.add(vaultMesh);
      scene.add(polyhouseGroup);

      // 5. Composite Solar ESP32 Sensor Towers (Shared Geometries & Materials)
      const sensorTowersGroup = new THREE.Group();
      const towerStatusMats: Record<string, THREE.MeshStandardMaterial> = {};

      const sharedFlangeGeo = new THREE.CylinderGeometry(0.28, 0.34, 0.12, 12);
      const sharedFlangeMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        roughness: 0.7,
        metalness: 0.4,
      });
      const sharedMastGeo = new THREE.CylinderGeometry(0.055, 0.065, 1.85, 10);
      const sharedMastMat = new THREE.MeshStandardMaterial({
        color: 0xcbd5e1,
        roughness: 0.3,
        metalness: 0.75,
      });
      const sharedBoxGeo = new THREE.BoxGeometry(0.36, 0.44, 0.26);
      const sharedBoxMat = new THREE.MeshStandardMaterial({
        color: 0xf1f5f9,
        roughness: 0.4,
        metalness: 0.2,
      });
      const sharedSolarGeo = new THREE.BoxGeometry(0.68, 0.04, 0.48);
      const sharedSolarMat = new THREE.MeshStandardMaterial({
        color: 0x1e3a5f,
        roughness: 0.22,
        metalness: 0.65,
      });
      const sharedAntennaGeo = new THREE.CylinderGeometry(0.015, 0.02, 0.55, 6);
      const sharedAntennaMat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        roughness: 0.4,
        metalness: 0.6,
      });
      const sharedCollarGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.14, 12);
      const sharedProxyGeo = new THREE.BoxGeometry(0.95, 2.4, 0.95);
      const sharedProxyMat = new THREE.MeshBasicMaterial({ visible: false });

      TOWER_NODES_SPEC.forEach((spec) => {
        const gy = getGroundHeight(spec.x, spec.z) + 0.16;
        const tower = new THREE.Group();
        tower.position.set(spec.x, gy, spec.z);

        const baseFlange = new THREE.Mesh(sharedFlangeGeo, sharedFlangeMat);
        baseFlange.position.y = 0.06;
        tower.add(baseFlange);

        const mast = new THREE.Mesh(sharedMastGeo, sharedMastMat);
        mast.position.y = 0.98;
        tower.add(mast);

        const box = new THREE.Mesh(sharedBoxGeo, sharedBoxMat);
        box.position.set(0, 1.15, 0.12);
        tower.add(box);

        const solarPanel = new THREE.Mesh(sharedSolarGeo, sharedSolarMat);
        solarPanel.position.set(0, 1.94, 0);
        solarPanel.rotation.x = 0.38;
        tower.add(solarPanel);

        const antenna = new THREE.Mesh(sharedAntennaGeo, sharedAntennaMat);
        antenna.position.set(0.18, 2.15, -0.1);
        tower.add(antenna);

        const statusCollarMat = new THREE.MeshStandardMaterial({
          color: 0x4a8c55,
          emissive: 0x184d23,
          roughness: 0.25,
          metalness: 0.3,
        });
        towerStatusMats[spec.id] = statusCollarMat;
        const statusCollar = new THREE.Mesh(sharedCollarGeo, statusCollarMat);
        statusCollar.position.set(0, 1.58, 0);
        tower.add(statusCollar);

        const hitProxy = new THREE.Mesh(sharedProxyGeo, sharedProxyMat);
        hitProxy.position.y = 1.2;
        hitProxy.layers.set(1);
        hitProxy.userData = {
          type: 'device',
          deviceId: spec.id,
          zoneId: spec.zoneId,
          label: `${spec.id} · ${spec.name}`,
        };
        tower.add(hitProxy);
        clickableObjects.push(hitProxy);

        sensorTowersGroup.add(tower);
      });
      scene.add(sensorTowersGroup);

      // 6. InstancedMesh 3D Crop Canopy Across the 4 Farm Plots
      const cropCount = 220;
      const stalkGeo = new THREE.ConeGeometry(0.16, 0.65, 5);
      stalkGeo.translate(0, 0.325, 0);
      const cropMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.65,
        metalness: 0.05,
      });
      const cropInstanced = new THREE.InstancedMesh(stalkGeo, cropMat, cropCount);

      const cropBaseTransforms: { x: number; y: number; z: number; scale: number; rotY: number }[] = [];
      let idx = 0;
      plotDefs.forEach((p) => {
        const rows = 5;
        const cols = 10;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            if (idx >= cropCount) break;
            const cx = p.x - p.w * 0.42 + (c / (cols - 1)) * (p.w * 0.84);
            const cz = p.z - p.d * 0.4 + (r / (rows - 1)) * (p.d * 0.8);
            if (Math.hypot(cx - p.x, cz - p.z) < 0.75) continue;
            const cy = getGroundHeight(cx, cz) + 0.16;
            const scaleVar = 0.85 + ((idx * 19) % 30) / 100;
            const rotY = (idx * 1.1) % Math.PI;
            cropBaseTransforms.push({ x: cx, y: cy, z: cz, scale: scaleVar, rotY });

            _dummyObj.position.set(cx, cy, cz);
            _dummyObj.rotation.set(0, rotY, 0);
            _dummyObj.scale.set(scaleVar, scaleVar, scaleVar);
            _dummyObj.updateMatrix();
            cropInstanced.setMatrixAt(idx, _dummyObj.matrix);

            _tempColor.setHex(
              p.id === 'ZONE-B' ? 0x3b8c52 : p.id === 'ZONE-C' ? 0x769442 : 0x4a8c55
            );
            cropInstanced.setColorAt(idx, _tempColor);
            idx++;
          }
        }
      });
      cropInstanced.count = cropBaseTransforms.length;
      cropInstanced.instanceMatrix.needsUpdate = true;
      if (cropInstanced.instanceColor) cropInstanced.instanceColor.needsUpdate = true;
      cropInstanced.computeBoundingSphere();
      scene.add(cropInstanced);

      // 7. Irrigation Pipelines & Animated Flow Particles
      const irrigationGroup = new THREE.Group();
      const pipeMat = new THREE.MeshStandardMaterial({
        color: 0x3b82a6,
        roughness: 0.3,
        metalness: 0.6,
      });

      plotDefs.forEach((p) => {
        const curve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, getGroundHeight(0, 0) + 0.25, 0),
          new THREE.Vector3(p.x * 0.5, getGroundHeight(p.x * 0.5, p.z * 0.5) + 0.2, p.z * 0.5),
          new THREE.Vector3(p.x, getGroundHeight(p.x, p.z) + 0.2, p.z),
        ]);
        const tubeGeo = new THREE.TubeGeometry(curve, 14, 0.055, 6, false);
        const tubeMesh = new THREE.Mesh(tubeGeo, pipeMat);
        irrigationGroup.add(tubeMesh);
      });

      const dropletCount = 40;
      const dropletGeo = new THREE.SphereGeometry(0.085, 6, 6);
      const dropletMat = new THREE.MeshBasicMaterial({ color: 0x67c3f3 });
      const dropletInstanced = new THREE.InstancedMesh(dropletGeo, dropletMat, dropletCount);
      dropletInstanced.frustumCulled = false;
      irrigationGroup.add(dropletInstanced);
      scene.add(irrigationGroup);

      // 8. Natural Perimeter Windbreak Trees & Rocks
      const perimeterGroup = new THREE.Group();
      const rockGeo = new THREE.DodecahedronGeometry(0.45, 0);
      const rockMat = new THREE.MeshStandardMaterial({
        color: 0x4a524a,
        roughness: 0.9,
        metalness: 0.05,
      });
      const trunkGeo = new THREE.CylinderGeometry(0.12, 0.18, 0.9, 6);
      const trunkMat = new THREE.MeshStandardMaterial({ color: 0x4a3525, roughness: 0.85 });
      const crown1Geo = new THREE.ConeGeometry(0.75, 1.5, 7);
      const crown2Geo = new THREE.ConeGeometry(0.55, 1.2, 7);
      const foliageMat = new THREE.MeshStandardMaterial({ color: 0x1e462c, roughness: 0.75 });

      const perimeterCoords = [
        [-12.5, -10.5],
        [-8.5, -11.2],
        [0, -11.5],
        [8.5, -11.0],
        [12.5, -10.2],
        [-13.0, 0],
        [13.0, 0],
        [-12.2, 10.2],
        [0, 11.2],
        [12.2, 10.4],
      ];
      perimeterCoords.forEach(([px, pz], i) => {
        const py = getGroundHeight(px, pz);
        const tree = new THREE.Group();
        tree.position.set(px, py, pz);
        const trunk = new THREE.Mesh(trunkGeo, trunkMat);
        trunk.position.y = 0.45;
        tree.add(trunk);
        const crown1 = new THREE.Mesh(crown1Geo, foliageMat);
        crown1.position.y = 1.3;
        tree.add(crown1);
        const crown2 = new THREE.Mesh(crown2Geo, foliageMat);
        crown2.position.y = 2.0;
        tree.add(crown2);
        const s = 0.9 + (i % 3) * 0.2;
        tree.scale.setScalar(s);
        perimeterGroup.add(tree);

        const rock = new THREE.Mesh(rockGeo, rockMat);
        rock.position.set(px + 0.9, getGroundHeight(px + 0.9, pz - 0.4) + 0.2, pz - 0.4);
        rock.scale.set(1.2, 0.7, 1.0);
        perimeterGroup.add(rock);
      });
      scene.add(perimeterGroup);

      // Raycaster on Layer 1
      const raycaster = new THREE.Raycaster();
      raycaster.layers.set(1);
      const pointer = new THREE.Vector2();

      const handleCanvasClick = (event: MouseEvent) => {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);

        const hits = raycaster.intersectObjects(clickableObjects, false);
        if (hits.length > 0) {
          const hitObj = hits[0].object;
          const uData = hitObj.userData;
          if (uData?.type === 'device' && uData.deviceId) {
            const found = stateRef.current.deviceById[uData.deviceId];
            if (found) {
              setInspectedLabel(
                `${found.id} · ${found.name} (${found.readings.soilMoisturePct}% VWC)`
              );
              stateRef.current.onSelectDevice?.(found);
            } else if (uData.label) {
              setInspectedLabel(uData.label);
            }
          } else if (uData?.label) {
            setInspectedLabel(uData.label);
          }

          if (uData?.zoneId && CAMERA_PRESETS[uData.zoneId as CameraPresetKey]) {
            setCameraPreset(uData.zoneId as CameraPresetKey);
            stateRef.current.onSelectZone?.(uData.zoneId);
          }
        }
      };

      renderer.domElement.addEventListener('click', handleCanvasClick);

      const handleResize = () => {
        if (!mountEl) return;
        const w = mountEl.clientWidth || 800;
        const h = mountEl.clientHeight || 430;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };

      const resizeObs = new ResizeObserver(() => {
        requestAnimationFrame(handleResize);
      });
      resizeObs.observe(mountEl);

      // Pause WebGL render loop automatically when offscreen or tab hidden
      let isVisibleInViewport = true;
      const intersectObs = new IntersectionObserver(
        (entries) => {
          if (entries[0]) {
            isVisibleInViewport = entries[0].isIntersecting;
          }
        },
        { threshold: 0.05 }
      );
      intersectObs.observe(mountEl);

      const clock = new THREE.Clock();
      let animFrameId = 0;
      let activePresetTrack: CameraPresetKey | null = 'overview';

      const onControlStart = () => {
        activePresetTrack = null;
      };
      controls.addEventListener('start', onControlStart);

      let lastPresetWatched: CameraPresetKey = stateRef.current.cameraPreset;
      let lastAppliedNdviFactor = -1;

      const animate = () => {
        animFrameId = requestAnimationFrame(animate);
        if (!isVisibleInViewport || document.hidden) return;

        const delta = Math.min(clock.getDelta(), 0.1);
        const elapsed = clock.getElapsedTime();
        const st = stateRef.current;

        if (st.cameraPreset !== lastPresetWatched) {
          lastPresetWatched = st.cameraPreset;
          activePresetTrack = st.cameraPreset;
        }

        if (activePresetTrack) {
          const targetDef = CAMERA_PRESETS[activePresetTrack];
          _camTargetPos.set(...targetDef.pos);
          _camLookTarget.set(...targetDef.lookAt);

          camera.position.x = THREE.MathUtils.damp(
            camera.position.x,
            _camTargetPos.x,
            4.5,
            delta
          );
          camera.position.y = THREE.MathUtils.damp(
            camera.position.y,
            _camTargetPos.y,
            4.5,
            delta
          );
          camera.position.z = THREE.MathUtils.damp(
            camera.position.z,
            _camTargetPos.z,
            4.5,
            delta
          );

          controls.target.x = THREE.MathUtils.damp(
            controls.target.x,
            _camLookTarget.x,
            5.0,
            delta
          );
          controls.target.y = THREE.MathUtils.damp(
            controls.target.y,
            _camLookTarget.y,
            5.0,
            delta
          );
          controls.target.z = THREE.MathUtils.damp(
            controls.target.z,
            _camLookTarget.z,
            5.0,
            delta
          );

          if (camera.position.distanceTo(_camTargetPos) < 0.06) {
            activePresetTrack = null;
          }
        }

        controls.update();

        // Update layer visibilities
        sensorTowersGroup.visible = st.showSensorTowers;
        cropInstanced.visible = st.showCropCanopy;
        irrigationGroup.visible = st.showIrrigationFlow;

        // Sync tower beacon colors with live device status
        if (st.showSensorTowers) {
          for (let i = 0; i < TOWER_NODES_SPEC.length; i++) {
            const nodeId = TOWER_NODES_SPEC[i].id;
            const collarMat = towerStatusMats[nodeId];
            const dev = st.deviceById[nodeId];
            if (collarMat && dev) {
              const isStale = dev.status === 'stale';
              collarMat.color.setHex(isStale ? 0xdca849 : 0x4a8c55);
              collarMat.emissive.setHex(isStale ? 0x6b4c12 : 0x184d23);
            }
          }
        }

        // Update reservoir water sight-glass height from live tankLevelPct
        const tankScaleY = Math.max(0.15, Math.min(1.0, st.tankLevelPct / 100));
        waterColumnMesh.scale.y = THREE.MathUtils.damp(
          waterColumnMesh.scale.y,
          tankScaleY,
          4,
          delta
        );

        // Update plot moisture colors using O(1) lookup map
        const activeDayPoint =
          st.twinSeries && st.twinSeries[st.selectedDayIndex]
            ? st.twinSeries[st.selectedDayIndex]
            : null;

        for (let i = 0; i < plotDefs.length; i++) {
          const p = plotDefs[i];
          const mat = plotMats[p.id];
          if (!mat) continue;
          let moisture = st.moistureByZone[p.id] ?? 45;
          if (activeDayPoint) {
            moisture =
              p.id === 'ZONE-B'
                ? activeDayPoint.predictedMoisturePct - 6
                : activeDayPoint.predictedMoisturePct;
          }

          if (st.showMoistureHeatmap) {
            const norm = THREE.MathUtils.clamp((moisture - 25) / 40, 0, 1);
            if (norm < 0.5) {
              _tempColor.copy(_colorDry).lerp(_colorOptimal, norm * 2);
            } else {
              _tempColor.copy(_colorOptimal).lerp(_colorWet, (norm - 0.5) * 2);
            }
            mat.color.lerp(_tempColor, 0.15);
          } else {
            _tempColor.setHex(p.color);
            mat.color.lerp(_tempColor, 0.15);
          }
        }

        // Only upload 220 crop instance matrices when NDVI factor changes (Zero redundant GPU buffer uploads per frame)
        if (st.showCropCanopy) {
          const ndviFactor = activeDayPoint
            ? Number((activeDayPoint.cropNdvi / 0.72).toFixed(3))
            : 1.0;
          if (Math.abs(ndviFactor - lastAppliedNdviFactor) > 0.002) {
            lastAppliedNdviFactor = ndviFactor;
            for (let i = 0; i < cropBaseTransforms.length; i++) {
              const item = cropBaseTransforms[i];
              _dummyObj.position.set(item.x, item.y, item.z);
              _dummyObj.rotation.set(0, item.rotY, 0);
              const sy = item.scale * ndviFactor;
              _dummyObj.scale.set(item.scale, sy, item.scale);
              _dummyObj.updateMatrix();
              cropInstanced.setMatrixAt(i, _dummyObj.matrix);
            }
            cropInstanced.instanceMatrix.needsUpdate = true;
          }
        }

        // Animate irrigation flow droplets using O(1) valve lookup
        if (st.showIrrigationFlow) {
          for (let i = 0; i < dropletCount; i++) {
            const plotIdx = i & 3; // Fast modulo 4
            const p = plotDefs[plotIdx];
            const isActive = st.valveActiveByZone[p.id] ?? false;
            const speed = isActive ? 0.55 : 0.18;
            const tProg = ((elapsed * speed + (i / dropletCount) * 4) % 1 + 1) % 1;
            const dx = p.x * tProg;
            const dz = p.z * tProg;
            const dy = getGroundHeight(dx, dz) + 0.24;
            _dummyObj.position.set(dx, dy, dz);
            _dummyObj.scale.setScalar(isActive ? 1.25 : 0.75);
            _dummyObj.updateMatrix();
            dropletInstanced.setMatrixAt(i, _dummyObj.matrix);
          }
          dropletInstanced.instanceMatrix.needsUpdate = true;
        }

        renderer.render(scene, camera);
      };

      animate();

      return () => {
        cancelAnimationFrame(animFrameId);
        resizeObs.disconnect();
        intersectObs.disconnect();
        renderer.domElement.removeEventListener('click', handleCanvasClick);
        controls.dispose();
        soilTexture.dispose();
        terrainGeo.dispose();
        terrainMat.dispose();
        sharedBedGeo.dispose();
        stalkGeo.dispose();
        cropMat.dispose();
        dropletGeo.dispose();
        dropletMat.dispose();
        renderer.dispose();
      };
    }, [viewMode]);

    const activeDayPoint =
      twinSeries && twinSeries[selectedDayIndex] ? twinSeries[selectedDayIndex] : null;

    return (
      <div
        className={`relative w-full ${heightClass} rounded-xl overflow-hidden border border-[var(--agri-border)] bg-[#0D1912] text-white select-none`}
      >
        {/* Top-Left Minimal Spec & Telemetry Badge */}
        <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-2 pointer-events-none">
          <div className="px-3 py-1.5 rounded-lg bg-[#0B140E]/95 border border-white/15 text-xs font-mono-tabular flex items-center gap-2">
            <Compass className="w-3.5 h-3.5 text-[#5CA668]" />
            <span className="font-semibold text-white">{inspectedLabel}</span>
            {activeDayPoint && (
              <>
                <span className="text-white/40">·</span>
                <span className="text-[#55A3D1]">
                  {activeDayPoint.dateShort} · NDVI {formatNumber(activeDayPoint.cropNdvi, lang)}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Top-Right 3D / 2D Mode Switch */}
        <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5">
          {webglSupported && (
            <button
              type="button"
              onClick={() => setViewMode((m) => (m === '3d' ? '2d' : '3d'))}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#0B140E]/95 hover:bg-[#173D2B] border border-white/15 text-[11px] font-medium text-white cursor-pointer"
            >
              {viewMode === '3d' ? (
                <>
                  <MapIcon className="w-3.5 h-3.5 text-[#5CA668]" />
                  <span>{t('common:hud3d.mode2D')}</span>
                </>
              ) : (
                <>
                  <Box className="w-3.5 h-3.5 text-[#5CA668]" />
                  <span>{t('common:hud3d.mode3D')}</span>
                </>
              )}
            </button>
          )}
        </div>

        {/* Dedicated Isolated WebGL Canvas Container */}
        {viewMode === '3d' && webglSupported ? (
          <div ref={canvasMountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
        ) : (
          /* Interactive 2D Topological SVG Fallback Map */
          <div className="w-full h-full flex flex-col justify-between p-6 bg-[#0D1912]">
            <svg
              viewBox="0 0 720 340"
              className="w-full h-full max-h-[300px]"
              role="img"
              aria-label="2D Interactive Topological Farm Map"
            >
              <rect width="720" height="340" rx="12" fill="#0D1912" />
              <g
                onClick={() => {
                  setInspectedLabel('ZONE-A · Block A Paddy');
                  onSelectZone?.('ZONE-A');
                }}
                className="cursor-pointer"
              >
                <rect x="40" y="30" width="260" height="125" rx="8" fill="#1E4D36" stroke="#4A8C55" strokeWidth="2" />
                <text x="58" y="62" fill="#FFFFFF" fontSize="13" fontWeight="bold" fontFamily="monospace">
                  ZONE-A · BLOCK A PADDY (2.4 ha)
                </text>
                <text x="58" y="88" fill="#DCE8D5" fontSize="12" fontFamily="monospace">
                  ESP32-NODE-101 · Moisture: {devices[0]?.readings.soilMoisturePct ?? 54.2}%
                </text>
              </g>
              <g
                onClick={() => {
                  setInspectedLabel('ZONE-B · Block B Polyhouse');
                  onSelectZone?.('ZONE-B');
                }}
                className="cursor-pointer"
              >
                <rect x="420" y="30" width="260" height="125" rx="8" fill="#234838" stroke="#DCA849" strokeWidth="2" />
                <text x="438" y="62" fill="#FFFFFF" fontSize="13" fontWeight="bold" fontFamily="monospace">
                  ZONE-B · POLYHOUSE TOMATO (1.1 ha)
                </text>
                <text x="438" y="88" fill="#DCA849" fontSize="12" fontFamily="monospace">
                  ESP32-NODE-102 · Moisture: {devices[1]?.readings.soilMoisturePct ?? 33.8}%
                </text>
              </g>
              <g
                onClick={() => {
                  setInspectedLabel('ZONE-D · Block D Turmeric');
                  onSelectZone?.('ZONE-D');
                }}
                className="cursor-pointer"
              >
                <rect x="40" y="185" width="260" height="125" rx="8" fill="#2A3928" stroke="#DCA849" strokeDasharray="5 3" strokeWidth="2" />
                <text x="58" y="218" fill="#FFFFFF" fontSize="13" fontWeight="bold" fontFamily="monospace">
                  ZONE-D · TURMERIC LOAM (1.5 ha)
                </text>
                <text x="58" y="244" fill="#DCA849" fontSize="12" fontFamily="monospace">
                  ESP32-NODE-104 · STALE (47m ago)
                </text>
              </g>
              <g
                onClick={() => {
                  setInspectedLabel('ZONE-C · Block C Millet');
                  onSelectZone?.('ZONE-C');
                }}
                className="cursor-pointer"
              >
                <rect x="420" y="185" width="260" height="125" rx="8" fill="#284A32" stroke="#4A8C55" strokeWidth="2" />
                <text x="438" y="218" fill="#FFFFFF" fontSize="13" fontWeight="bold" fontFamily="monospace">
                  ZONE-C · FINGER MILLET (1.8 ha)
                </text>
                <text x="438" y="244" fill="#DCE8D5" fontSize="12" fontFamily="monospace">
                  ESP32-NODE-103 · Moisture: {devices[3]?.readings.soilMoisturePct ?? 38.4}%
                </text>
              </g>
              <circle cx="360" cy="170" r="38" fill="#173D2B" stroke="#438EBA" strokeWidth="3" />
              <text x="328" y="166" fill="#FFFFFF" fontSize="10" fontWeight="bold" fontFamily="monospace">
                RPI-GW-01
              </text>
              <text x="330" y="182" fill="#55A3D1" fontSize="10" fontFamily="monospace">
                TANK {tankLevelPct}%
              </text>
            </svg>
          </div>
        )}

        {/* Collapsible Bottom Control Dock */}
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 w-[95%] max-w-[920px] flex flex-col items-center">
          <button
            type="button"
            onClick={() => setControlsCollapsed((c) => !c)}
            className="px-3 py-1 rounded-t-lg bg-[#0B140E] border border-b-0 border-white/15 text-[11px] font-mono-tabular text-[#DCE8D5] hover:text-white cursor-pointer flex items-center gap-1.5"
          >
            {controlsCollapsed ? (
              <>
                <Eye className="w-3 h-3 text-[#5CA668]" />
                <span>{t('common:hud3d.showControls')}</span>
              </>
            ) : (
              <>
                <EyeOff className="w-3 h-3 text-[#5CA668]" />
                <span>{t('common:hud3d.hideControls')}</span>
              </>
            )}
          </button>

          {!controlsCollapsed && (
            <div className="w-full rounded-xl bg-[#0B140E]/95 border border-white/15 px-3.5 py-2.5 space-y-2 shadow-xl">
              {twinSeries && twinSeries.length > 0 && onChangeDayIndex && (
                <div className="flex items-center gap-3 pb-2 border-b border-white/10 text-xs">
                  <span className="font-mono-tabular text-[#55A3D1] whitespace-nowrap">
                    {t('common:hud3d.timeScrubber')}:
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={twinSeries.length - 1}
                    value={selectedDayIndex}
                    onChange={(e) => onChangeDayIndex(Number(e.target.value))}
                    className="flex-1 accent-[#4A8C55] cursor-pointer"
                  />
                  <span className="font-mono-tabular text-white font-semibold whitespace-nowrap">
                    {twinSeries[selectedDayIndex]?.dateShort} ·{' '}
                    {formatNumber(twinSeries[selectedDayIndex]?.predictedMoisturePct ?? 50, lang)}% VWC
                  </span>
                </div>
              )}

              <div className="flex flex-wrap items-center justify-between gap-2">
                {/* Camera Focus Presets */}
                <div className="flex flex-wrap items-center gap-1">
                  {(
                    [
                      { id: 'overview', label: t('common:hud3d.cameraOverview') },
                      { id: 'ZONE-A', label: t('common:hud3d.cameraZoneA') },
                      { id: 'ZONE-B', label: t('common:hud3d.cameraZoneB') },
                      { id: 'ZONE-C', label: t('common:hud3d.cameraZoneC') },
                      { id: 'ZONE-D', label: t('common:hud3d.cameraZoneD') },
                      { id: 'ZONE-CENTRAL', label: t('common:hud3d.cameraReservoir') },
                    ] as const
                  ).map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => {
                        setCameraPreset(preset.id);
                        if (preset.id !== 'overview') {
                          onSelectZone?.(preset.id);
                        }
                      }}
                      className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors cursor-pointer whitespace-nowrap ${
                        cameraPreset === preset.id
                          ? 'bg-[#4A8C55] text-white font-semibold'
                          : 'bg-white/5 text-[#DCE8D5]/80 hover:bg-white/15 hover:text-white'
                      }`}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>

                {/* Layer Toggles */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setShowMoistureHeatmap((v) => !v)}
                    className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium cursor-pointer ${
                      showMoistureHeatmap
                        ? 'bg-[#2B729F] text-white'
                        : 'bg-white/5 text-white/60 hover:text-white'
                    }`}
                  >
                    <Layers className="w-3 h-3" />
                    <span>{t('common:hud3d.layerMoisture')}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setShowIrrigationFlow((v) => !v)}
                    className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium cursor-pointer ${
                      showIrrigationFlow
                        ? 'bg-[#3B82A6] text-white'
                        : 'bg-white/5 text-white/60 hover:text-white'
                    }`}
                  >
                    <Droplets className="w-3 h-3" />
                    <span>{t('common:hud3d.layerIrrigation')}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setShowCropCanopy((v) => !v)}
                    className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium cursor-pointer ${
                      showCropCanopy
                        ? 'bg-[#4A8C55] text-white'
                        : 'bg-white/5 text-white/60 hover:text-white'
                    }`}
                  >
                    <Sprout className="w-3 h-3" />
                    <span>{t('common:hud3d.layerCanopy')}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setShowSensorTowers((v) => !v)}
                    className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium cursor-pointer ${
                      showSensorTowers
                        ? 'bg-[#1E4D36] border border-[#5CA668]/40 text-white'
                        : 'bg-white/5 text-white/60 hover:text-white'
                    }`}
                  >
                    <Radio className="w-3 h-3" />
                    <span>{t('common:hud3d.layerSensors')}</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }
);
