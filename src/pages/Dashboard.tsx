/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState, useRef } from "react";
import { 
  Sprout, 
  Droplet, 
  Thermometer, 
  Activity, 
  AlertTriangle, 
  ArrowRight, 
  TrendingUp, 
  Cpu, 
  CloudSun,
  Wind,
  Flame,
  CheckCircle,
  HelpCircle,
  TrendingDown,
  Gauge,
  Wifi,
  CloudRain
} from "lucide-react";
import { SensorReading } from "../types";

interface DashboardProps {
  onNavigate: (page: string) => void;
  sensors: SensorReading[];
}

export default function Dashboard({ onNavigate, sensors }: DashboardProps) {
  const [metrics, setMetrics] = useState({
    avgMoisture: 38.5,
    avgTemp: 27.9,
    avgPH: 6.3,
    mlHealth: 98.4,
    nitrogen: 45,
    potassium: 42,
    phosphorus: 38
  });

  const [activeWeather, setActiveWeather] = useState({
    description: "Gentle Breeze",
    windSpeed: 4.2, // m/s
    humidity: 58,
    solarRad: 580 // W/m²
  });

  const [windIntensity, setWindIntensity] = useState<number>(1.0); // Multiplier for crop swaying
  const [activeWaterValve, setActiveWaterValve] = useState<boolean>(true); // For animated water flow override

  useEffect(() => {
    if (sensors.length > 0) {
      const sumMoisture = sensors.reduce((acc, s) => acc + s.soilMoisture, 0);
      const sumTemp = sensors.reduce((acc, s) => acc + s.temperature, 0);
      const sumPH = sensors.reduce((acc, s) => acc + s.pH, 0);
      const latest = sensors[0];
      setMetrics({
        avgMoisture: parseFloat((sumMoisture / sensors.length).toFixed(1)),
        avgTemp: parseFloat((sumTemp / sensors.length).toFixed(1)),
        avgPH: parseFloat((sumPH / sensors.length).toFixed(1)),
        mlHealth: 98.4,
        nitrogen: latest.nitrogen,
        potassium: latest.potassium,
        phosphorus: latest.phosphorus
      });
    }
  }, [sensors]);

  const latestAlerts = [
    {
      id: "alt-01",
      title: "Moisture Deficit in Soil Zone 4",
      desc: `FAO-56 Penman-Monteith water index dropped below target. Water reservoir valve recommended at 1250 L flushes.`,
      severity: "medium",
      action: "Optimize Irrigation",
      target: "irrigation"
    },
    {
      id: "alt-02",
      title: "Crop Suitability TabPFN Recommendation",
      desc: "Nitrogen accumulation is sub-optimal for maize. Switch to legumes or apply precision organic nitrogen.",
      severity: "info",
      action: "Check Suitability",
      target: "crop"
    }
  ];

  // Canvas Reference for the beautiful animated 3D farm simulation
  const mainCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animFrame: number;
    let tick = 0;

    // Drones state
    const drones = [
      { id: "drone-01", x: 0, y: 0, hoverHeight: 50, radius: 110, speed: 0.009, color: "#10b981", label: "AI Sentinel 1", angleOffset: 0 },
      { id: "drone-02", x: 0, y: 0, hoverHeight: 65, radius: 150, speed: -0.007, color: "#3b82f6", label: "AI Sentinel 2", angleOffset: Math.PI }
    ];

    // Rain/Water Flow particles on pipes
    const waterFlowParticles: Array<{ t: number; speed: number; yIndex: number }> = [
      { t: 0.0, speed: 0.015, yIndex: 0 },
      { t: 0.25, speed: 0.015, yIndex: 1 },
      { t: 0.5, speed: 0.015, yIndex: 2 },
      { t: 0.75, speed: 0.015, yIndex: 3 },
      { t: 0.1, speed: 0.018, yIndex: 1 },
      { t: 0.6, speed: 0.018, yIndex: 3 }
    ];

    // Background wind breeze curves
    const breezeLines = Array.from({ length: 6 }, (_, i) => ({
      y: 50 + i * 40,
      x: Math.random() * 800,
      speed: 2.5 + Math.random() * 3.5,
      length: 100 + Math.random() * 150
    }));

    const resizeAndDraw = () => {
      const container = canvas.parentElement;
      if (!container) return;
      canvas.width = container.clientWidth;
      canvas.height = 420;

      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      // --- 1. SOPIHSTICATED ATMOSPHERIC CHROMA GRADIENT ---
      // Natural Agriculture Harvest Sunrise theme
      const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
      bgGrad.addColorStop(0, "#f3f8f5");   // Fresh organic pale mint
      bgGrad.addColorStop(0.3, "#fef6eb"); // Soft harvest sun-glow cream
      bgGrad.addColorStop(0.7, "#fdf0e2"); // Sunny clay earth blush
      bgGrad.addColorStop(1, "#f3e7db");   // Solid warm clay ground
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Subtle tech background grids representation
      ctx.strokeStyle = "rgba(16, 185, 129, 0.04)";
      ctx.lineWidth = 1;
      for (let i = 0; i < h; i += 24) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(w, i);
        ctx.stroke();
      }

      // --- 2. THE SUNLIGHT GENERATOR ---
      const sunPulse = 45 + Math.sin(tick * 0.02) * 4;
      const sunX = w - 90;
      const sunY = 75;
      
      // Warm volumetric sun rays
      const sunRayGrad = ctx.createRadialGradient(sunX, sunY, 5, sunX, sunY, sunPulse * 2.5);
      sunRayGrad.addColorStop(0, "rgba(251, 191, 11, 0.25)");
      sunRayGrad.addColorStop(0.5, "rgba(251, 191, 11, 0.08)");
      sunRayGrad.addColorStop(1, "rgba(251, 191, 11, 0)");
      ctx.fillStyle = sunRayGrad;
      ctx.beginPath();
      ctx.arc(sunX, sunY, sunPulse * 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Sharp core of harvest sun
      ctx.fillStyle = "rgba(252, 211, 77, 0.6)";
      ctx.beginPath();
      ctx.arc(sunX, sunY, sunPulse, 0, Math.PI * 2);
      ctx.fill();

      // --- 3. AMBIENT WIND WINDSTREAKS & CLOUDS ---
      ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
      for (let c = 0; c < 3; c++) {
        const cx = ((tick * 0.12 + c * 260) % (w + 120)) - 120;
        const cy = 45 + c * 18;
        ctx.beginPath();
        ctx.arc(cx, cy, 15, 0, Math.PI * 2);
        ctx.arc(cx + 14, cy - 8, 20, 0, Math.PI * 2);
        ctx.arc(cx + 28, cy, 14, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fill();
      }

      // Wind Flow Streaks
      ctx.strokeStyle = "rgba(52, 211, 153, 0.22)";
      ctx.lineWidth = 2;
      breezeLines.forEach(line => {
        line.x += line.speed * windIntensity;
        if (line.x > w) {
          line.x = -line.length;
          line.y = 40 + Math.random() * 160;
        }
        ctx.beginPath();
        ctx.moveTo(line.x, line.y);
        ctx.lineTo(line.x + line.length, line.y - 12);
        ctx.stroke();
      });

      // --- 4. WIND POWER GENERATOR (3D WIND TURBINE) ---
      const wtX = w - 160;
      const wtY = 160;
      const wtH = 95;
      
      // Wind turbine polygonal base tower
      const wtGrad = ctx.createLinearGradient(wtX - 4, wtY, wtX + 4, wtY);
      wtGrad.addColorStop(0, "#cbd5e1");
      wtGrad.addColorStop(0.5, "#f1f5f9");
      wtGrad.addColorStop(1, "#94a3b8");
      
      ctx.fillStyle = wtGrad;
      ctx.beginPath();
      ctx.moveTo(wtX - 3, wtY + wtH);
      ctx.lineTo(wtX + 3, wtY + wtH);
      ctx.lineTo(wtX + 1, wtY);
      ctx.lineTo(wtX - 1, wtY);
      ctx.closePath();
      ctx.fill();

      // Nacelle (Head house)
      ctx.fillStyle = "#64748b";
      ctx.beginPath();
      ctx.ellipse(wtX, wtY, 6, 4, 0, 0, Math.PI * 2);
      ctx.fill();

      // Spinning aerodynamic rotor blades
      const windSpeedMultiplier = activeWeather.windSpeed;
      const wtAngle = tick * (windSpeedMultiplier * 0.012);
      
      ctx.fillStyle = "rgba(241, 245, 249, 0.95)";
      ctx.strokeStyle = "#cbd5e1";
      ctx.lineWidth = 1;
      for (let b = 0; b < 3; b++) {
        const angle = wtAngle + (b * Math.PI * 2) / 3;
        const bladeL = 40;
        const bladeW = 3.5;
        
        ctx.beginPath();
        // Draw elegant air-foil polygonal blade
        ctx.moveTo(wtX, wtY);
        // Tip of the blade
        const bx = wtX + Math.cos(angle) * bladeL;
        const by = wtY + Math.sin(angle) * bladeL;
        // Midpoint coordinates for width thickening
        const bmx = wtX + Math.cos(angle - 0.12) * (bladeL * 0.35);
        const bmy = wtY + Math.sin(angle - 0.12) * (bladeL * 0.35);
        
        ctx.lineTo(bmx - Math.sin(angle) * bladeW, bmy + Math.cos(angle) * bladeW);
        ctx.lineTo(bx, by);
        ctx.lineTo(bmx + Math.sin(angle) * bladeW, bmy - Math.cos(angle) * bladeW);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // --- 5. ISOMETRIC FARMLAND PARAMETERS ---
      const isoX = w / 2;
      const isoY = h / 2 - 25;
      const tileW = 56;
      const tileH = tileW / 2;
      const gridCount = 5; // 5x5 sub-division

      // Outer bounding coordinates of the 3D grid
      const farLeftX = isoX - gridCount * tileW;
      const farLeftY = isoY + gridCount * tileH;
      const farRightX = isoX + gridCount * tileW;
      const farRightY = isoY + gridCount * tileH;
      const peakX = isoX;
      const peakY = isoY;
      const lowX = isoX;
      const lowY = isoY + gridCount * tileH * 2;

      // --- 6. GEOLOGICAL SURFACE CUT-OUT (3D GROUND STRATA POLYGONS) ---
      // Draw LEFT geological wall
      const leftWallGrad = ctx.createLinearGradient(farLeftX, farLeftY, lowX, lowY);
      leftWallGrad.addColorStop(0, "#482816"); // Humus organic crust
      leftWallGrad.addColorStop(0.3, "#7c370c"); // Rich clay orange
      leftWallGrad.addColorStop(0.7, "#632709"); // Hard soil depth
      leftWallGrad.addColorStop(1, "#3c1002"); // Dusty bedrock level
      
      ctx.fillStyle = leftWallGrad;
      ctx.beginPath();
      ctx.moveTo(farLeftX, farLeftY);
      ctx.lineTo(lowX, lowY);
      ctx.lineTo(lowX, lowY + 75); // 75px depth
      ctx.lineTo(farLeftX, farLeftY + 75);
      ctx.closePath();
      ctx.fill();

      // Left Strata banding stripes
      ctx.fillStyle = "rgba(124, 55, 12, 0.45)"; // sub-clay bands
      ctx.beginPath();
      ctx.moveTo(farLeftX, farLeftY + 22);
      ctx.lineTo(lowX, lowY + 22);
      ctx.lineTo(lowX, lowY + 38);
      ctx.lineTo(farLeftX, farLeftY + 38);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = "rgba(43, 22, 10, 0.7)"; // stone gray bands at bottom
      ctx.beginPath();
      ctx.moveTo(farLeftX, farLeftY + 50);
      ctx.lineTo(lowX, lowY + 50);
      ctx.lineTo(lowX, lowY + 75);
      ctx.lineTo(farLeftX, farLeftY + 75);
      ctx.closePath();
      ctx.fill();

      // Embedded polygonal fossilized stones in Left Soil Profile
      ctx.fillStyle = "#8a7e72";
      ctx.strokeStyle = "#40362b";
      ctx.lineWidth = 1;
      const drawStone = (sx: number, sy: number, sz: number) => {
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + sz, sy - sz * 0.4);
        ctx.lineTo(sx + sz * 1.5, sy + sz * 0.2);
        ctx.lineTo(sx + sz * 0.8, sy + sz * 0.8);
        ctx.lineTo(sx - sz * 0.2, sy + sz * 0.5);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      };
      drawStone(farLeftX + 50, farLeftY + 55, 8);
      drawStone(farLeftX + 130, farLeftY + 65, 10);
      drawStone(farLeftX + 220, farLeftY + 74, 7);

      // Draw RIGHT geological wall
      const rightWallGrad = ctx.createLinearGradient(lowX, lowY, farRightX, farRightY);
      rightWallGrad.addColorStop(0, "#3a2012"); // Shaded Humus crust
      rightWallGrad.addColorStop(0.3, "#652b07"); // Clay-Orange
      rightWallGrad.addColorStop(0.7, "#501d04"); // Deep Soil
      rightWallGrad.addColorStop(1, "#280a01"); // Subsurface obsidian slate
      
      ctx.fillStyle = rightWallGrad;
      ctx.beginPath();
      ctx.moveTo(lowX, lowY);
      ctx.lineTo(farRightX, farRightY);
      ctx.lineTo(farRightX, farRightY + 75);
      ctx.lineTo(lowX, lowY + 75);
      ctx.closePath();
      ctx.fill();

      // Right Strata banding stripes
      ctx.fillStyle = "rgba(101, 43, 7, 0.4)";
      ctx.beginPath();
      ctx.moveTo(lowX, lowY + 18);
      ctx.lineTo(farRightX, farRightY + 18);
      ctx.lineTo(farRightX, farRightY + 34);
      ctx.lineTo(lowX, lowY + 34);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = "rgba(22, 10, 5, 0.65)";
      ctx.beginPath();
      ctx.moveTo(lowX, lowY + 52);
      ctx.lineTo(farRightX, farRightY + 52);
      ctx.lineTo(farRightX, farRightY + 75);
      ctx.lineTo(lowX, lowY + 75);
      ctx.closePath();
      ctx.fill();

      drawStone(lowX + 80, lowY + 68, 9);
      drawStone(lowX + 180, lowY + 54, 7);

      // --- 7. ECO-CANAL IRRIGATION WATER ditch (Along front edge) ---
      // We will place an open flowing stream right next to bottom corner of farmland
      const canalW = 12;
      ctx.fillStyle = "rgba(37, 99, 235, 0.15)";
      ctx.beginPath();
      ctx.moveTo(farLeftX - canalW, farLeftY);
      ctx.lineTo(lowX - canalW, lowY);
      ctx.lineTo(farRightX + canalW, farRightY);
      ctx.lineTo(farRightX, farRightY);
      ctx.lineTo(lowX, lowY);
      ctx.lineTo(farLeftX, farLeftY);
      ctx.closePath();
      ctx.fill();

      // Draw real geometric water polygon for flowing irrigation river
      const flowOffset = (tick * 1.5) % 80;
      ctx.fillStyle = "rgba(14, 165, 233, 0.75)"; // Light reflecting stream sky blue
      ctx.beginPath();
      ctx.moveTo(farLeftX, farLeftY + 2);
      ctx.lineTo(lowX, lowY + 2);
      ctx.lineTo(farRightX, farRightY + 2);
      ctx.lineTo(farRightX + 4, farRightY + 7);
      ctx.lineTo(lowX, lowY + 8);
      ctx.lineTo(farLeftX - 4, farLeftY + 7);
      ctx.closePath();
      ctx.fill();

      // Glowing moving ripple markers within the canal
      ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
      ctx.lineWidth = 1.2;
      for (let wLine = 0; wLine < 5; wLine++) {
        const wPos = (flowOffset + wLine * 60) % (w - 40);
        ctx.beginPath();
        ctx.moveTo(wPos - 40, lowY + 4 + Math.sin(tick * 0.08 + wLine) * 1.5);
        ctx.lineTo(wPos, lowY + 3 + Math.sin(tick * 0.08 + wLine) * 1.5);
        ctx.stroke();
      }

      // --- 8. RENDER 5X5 INTERACTIVE SOIL GRID & CROPS ---
      for (let x = 0; x < gridCount; x++) {
        for (let y = 0; y < gridCount; y++) {
          const screenX = isoX + (x - y) * tileW;
          const screenY = isoY + (x + y) * tileH;

          // Compute custom moisture visual shaders
          const moistureVal = metrics.avgMoisture;
          let groundFill = "#854d0e"; // standard rich tilled ground
          let borderStroke = "#e2e8f0";

          if (moistureVal > 40) {
            groundFill = "#451a03"; // super hydration deep compost humus
            borderStroke = "rgba(16, 185, 129, 0.4)";
          } else if (moistureVal > 25) {
            groundFill = "#78350f"; // ideal farming loamy loam
            borderStroke = "rgba(16, 185, 129, 0.2)";
          } else {
            groundFill = "#a16207"; // dry sandy loam
            borderStroke = "rgba(239, 68, 68, 0.45)";
          }

          // Individual farm grid polygon base
          ctx.fillStyle = groundFill;
          ctx.strokeStyle = borderStroke;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(screenX, screenY);
          ctx.lineTo(screenX + tileW, screenY + tileH);
          ctx.lineTo(screenX, screenY + tileH * 2);
          ctx.lineTo(screenX - tileW, screenY + tileH);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          // Organic crop tilling lines inside the polygon (fine tilled rows)
          ctx.strokeStyle = "rgba(0,0,0,0.18)";
          ctx.lineWidth = 1;
          for (let row = -2; row <= 2; row++) {
            ctx.beginPath();
            ctx.moveTo(screenX - tileW + (row + 2) * 10, screenY + tileH - (row + 2) * 5);
            ctx.lineTo(screenX + (row + 2) * 10, screenY + tileH * 2 - (row + 2) * 5);
            ctx.stroke();
          }

          // Sway mechanics
          const swayAmt = Math.sin(tick * 0.05 + x * 0.45 + y * 0.35) * 6 * windIntensity;
          ctx.lineWidth = 2;

          const plantX = screenX;
          const plantY = screenY + tileH;

          // Plot distribution patterns of custom modular crop categories
          if (x % 2 === 0 && y % 2 === 0) {
            // --- MAIZE VEGETATION STEMS ---
            // Draw dual green stalks
            ctx.strokeStyle = "#047857";
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.moveTo(plantX - 2, plantY);
            ctx.quadraticCurveTo(plantX - 2 + swayAmt * 0.5, plantY - 10, plantX - 2 + swayAmt, plantY - 24);
            ctx.moveTo(plantX + 2, plantY);
            ctx.quadraticCurveTo(plantX + 2 + swayAmt * 0.3, plantY - 8, plantX + swayAmt * 0.9, plantY - 20);
            ctx.stroke();

            // Broad leafy polygons
            ctx.fillStyle = "#10b981";
            ctx.beginPath();
            // Stretchy leaf polygon 1
            ctx.moveTo(plantX - 2 + swayAmt * 0.5, plantY - 10);
            ctx.quadraticCurveTo(plantX - 12 + swayAmt, plantY - 14, plantX - 20 + swayAmt, plantY - 8);
            ctx.quadraticCurveTo(plantX - 8 + swayAmt, plantY - 6, plantX - 2 + swayAmt * 0.5, plantY - 10);
            // Stretchy leaf polygon 2
            ctx.moveTo(plantX + 2 + swayAmt * 0.3, plantY - 8);
            ctx.quadraticCurveTo(plantX + 14 + swayAmt, plantY - 12, plantX + 22 + swayAmt, plantY - 6);
            ctx.quadraticCurveTo(plantX + 8 + swayAmt, plantY - 4, plantX + 2 + swayAmt * 0.3, plantY - 8);
            ctx.closePath();
            ctx.fill();

            // Ripe golden corn cobs
            ctx.fillStyle = "#facc15";
            ctx.beginPath();
            ctx.ellipse(plantX - 3 + swayAmt * 0.6, plantY - 14, 2.5, 4.5, Math.PI / 6, 0, Math.PI * 2);
            ctx.ellipse(plantX + 4 + swayAmt * 0.5, plantY - 11, 2, 4, -Math.PI / 6, 0, Math.PI * 2);
            ctx.fill();

            // Tiny corn husk wrap
            ctx.strokeStyle = "#0d9488";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(plantX - 3 + swayAmt * 0.6, plantY - 10, 3, 0, Math.PI, true);
            ctx.stroke();

          } else if (y === 1 || x === 3) {
            // --- ORGANIC WHEAT CLUSTERS ---
            ctx.strokeStyle = "#b45309"; // rich seed husk
            ctx.lineWidth = 1.3;
            
            for (let stalkIdx = -1; stalkIdx <= 1; stalkIdx++) {
              const offX = stalkIdx * 4;
              const sSway = swayAmt * (0.8 + stalkIdx * 0.1);
              const tx = plantX + offX + sSway;
              const ty = plantY - 16 - Math.abs(stalkIdx) * 2;
              
              ctx.beginPath();
              ctx.moveTo(plantX + offX, plantY);
              ctx.quadraticCurveTo(plantX + offX + sSway * 0.5, plantY - 9, tx, ty);
              ctx.stroke();

              // Draw beaded polygonal wheat beard heads (grain nodes)
              ctx.fillStyle = "#f59e0b"; // Rich ambers
              for (let node = 0; node < 4; node++) {
                const ny = ty + node * 3;
                const nx = tx - (swayAmt * 0.15) * (node * 0.1);
                ctx.beginPath();
                ctx.arc(nx, ny, 1.6, 0, Math.PI * 2);
                ctx.fill();
                
                // Fine golden hair filaments
                ctx.strokeStyle = "rgba(245, 158, 11, 0.65)";
                ctx.lineWidth = 0.6;
                ctx.beginPath();
                ctx.moveTo(nx, ny);
                ctx.lineTo(nx - 3, ny - 3);
                ctx.moveTo(nx, ny);
                ctx.lineTo(nx + 3, ny - 3);
                ctx.stroke();
              }
            }

          } else {
            // --- SUSTAINABLE VALUABLE COTTON PLANTS ---
            // Draw woody branching stalks
            ctx.strokeStyle = "#7c2d12";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(plantX, plantY);
            ctx.quadraticCurveTo(plantX + swayAmt * 0.4, plantY - 6, plantX + swayAmt * 0.7, plantY - 14);
            ctx.stroke();

            // Puffy polygonal cotton bolls (fluffy cloud spheres)
            ctx.fillStyle = "#ffffff";
            ctx.strokeStyle = "#e2e8f0";
            ctx.lineWidth = 0.8;
            
            const bX = plantX + swayAmt * 0.7;
            const bY = plantY - 14;
            
            // Render 3 overlapping cotton spheres
            ctx.beginPath();
            ctx.arc(bX - 2.5, bY, 3, 0, Math.PI * 2);
            ctx.arc(bX + 2.5, bY, 3, 0, Math.PI * 2);
            ctx.arc(bX, bY - 3, 3.5, 0, Math.PI * 2);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            // Defensive green sepals underneath
            ctx.fillStyle = "#4d7c0f";
            ctx.beginPath();
            ctx.moveTo(bX - 4, bY + 1.5);
            ctx.lineTo(bX, bY + 5);
            ctx.lineTo(bX + 4, bY + 1.5);
            ctx.closePath();
            ctx.fill();
          }

          // Active Smart IoT probes and soil signal arrays
          if (x === 2 && y === 2) {
            // Render structured metallic IoT probe with red indicator lights
            ctx.strokeStyle = "#475569";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(plantX, plantY + 4);
            ctx.lineTo(plantX, plantY - 12);
            ctx.stroke();

            // Probe control box polygon
            ctx.fillStyle = "#f8fafc";
            ctx.strokeStyle = "#1e293b";
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.rect(plantX - 4, plantY - 18, 8, 6);
            ctx.fill();
            ctx.stroke();

            // Flashing LED diodes
            ctx.fillStyle = (tick % 30 < 15) ? "#dc2626" : "#22c55e"; // Flashing red/green
            ctx.beginPath();
            ctx.arc(plantX, plantY - 15, 1.5, 0, Math.PI * 2);
            ctx.fill();

            // Render digital ping telemetry circles
            const pingRadius = (tick % 45) * 0.8;
            ctx.strokeStyle = `rgba(34, 197, 94, ${Math.max(0, 1 - (pingRadius / 36))})`;
            ctx.lineWidth = 1.1;
            ctx.beginPath();
            ctx.arc(plantX, plantY - 15, Math.max(0.1, pingRadius), 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }

      // --- 9. GEOMETRIC RED BARN FARMHOUSE (3D COMPLEX POLYGONS) ---
      // Placed on the back-right edge of the farm scene
      const barnX = isoX + 130;
      const barnY = isoY + 25;
      
      // LEFT FACE OF BARN WALL (Shadow side polygon)
      ctx.fillStyle = "#991b1b"; // Deep shaded barn red
      ctx.beginPath();
      ctx.moveTo(barnX, barnY);
      ctx.lineTo(barnX - 25, barnY - 12);
      ctx.lineTo(barnX - 25, barnY - 35);
      ctx.lineTo(barnX, barnY - 22);
      ctx.closePath();
      ctx.fill();

      // FRONT FACE OF BARN WALL (Lit side polygon)
      ctx.fillStyle = "#dc2626"; // Vibrant primary crimson
      ctx.beginPath();
      ctx.moveTo(barnX, barnY);
      ctx.lineTo(barnX + 30, barnY + 15);
      ctx.lineTo(barnX + 30, barnY - 10);
      ctx.lineTo(barnX, barnY - 22);
      ctx.closePath();
      ctx.fill();

      // GABLE END TRIGON (Above the front wall)
      ctx.fillStyle = "#b91c1c";
      ctx.beginPath();
      ctx.moveTo(barnX, barnY - 22);
      ctx.lineTo(barnX + 30, barnY - 10);
      const bPeakX = barnX + 15;
      const bPeakY = barnY - 32;
      ctx.lineTo(bPeakX, bPeakY);
      ctx.closePath();
      ctx.fill();

      // GAMBREL ROOF SURFACE SLOP 1 (Front Side roof)
      ctx.fillStyle = "#4b5563"; // Dark charcoal slate
      ctx.beginPath();
      ctx.moveTo(barnX + 30, barnY - 10);
      ctx.lineTo(bPeakX, bPeakY);
      ctx.lineTo(bPeakX - 25, bPeakY - 12);
      ctx.lineTo(barnX - 25, barnY - 35);
      ctx.closePath();
      ctx.fill();

      // GAMBREL ROOF SURFACE SLOP 2 (Lit ridge roof)
      ctx.fillStyle = "#6b7280";
      ctx.beginPath();
      ctx.moveTo(bPeakX, bPeakY);
      ctx.lineTo(bPeakX - 25, bPeakY - 12);
      ctx.lineTo(bPeakX - 25, bPeakY - 18);
      ctx.lineTo(bPeakX, bPeakY - 6);
      ctx.closePath();
      ctx.fill();

      // Traditional white cross barn doors (X trim)
      ctx.fillStyle = "#7f1d1d";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      // Left Door panel
      ctx.beginPath();
      ctx.moveTo(barnX + 6, barnY + 3);
      ctx.lineTo(barnX + 13, barnY + 6);
      ctx.lineTo(barnX + 13, barnY - 6);
      ctx.lineTo(barnX + 6, barnY - 9);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // Draw standard "X" lines
      ctx.beginPath();
      ctx.moveTo(barnX + 6, barnY + 3);
      ctx.lineTo(barnX + 13, barnY - 6);
      ctx.moveTo(barnX + 13, barnY + 6);
      ctx.lineTo(barnX + 6, barnY - 9);
      ctx.stroke();

      // Right Door panel
      ctx.beginPath();
      ctx.moveTo(barnX + 16, barnY + 8);
      ctx.lineTo(barnX + 23, barnY + 11);
      ctx.lineTo(barnX + 23, barnY - 1);
      ctx.lineTo(barnX + 16, barnY - 4);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // Draw standard "X" lines
      ctx.beginPath();
      ctx.moveTo(barnX + 16, barnY + 8);
      ctx.lineTo(barnX + 23, barnY - 1);
      ctx.moveTo(barnX + 23, barnY + 11);
      ctx.lineTo(barnX + 16, barnY - 4);
      ctx.stroke();

      // Circular Loft window glowing with orange warm light
      ctx.fillStyle = "#f59e0b";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(bPeakX, bPeakY + 9, 3.5, 4.5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // --- 10. HIGH TECH ECO-GREENHOUSE (TRANSLUCENT POLYGONS) ---
      // Placed on far-left background
      const ghX = isoX - 140;
      const ghY = isoY + 80;
      const ghW = 32;
      const ghH = 18;

      // Greenhouse Base block brick wall
      ctx.fillStyle = "#d1fae5";
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(ghX, ghY);
      ctx.lineTo(ghX + ghW, ghY + ghW * 0.5);
      ctx.lineTo(ghX, ghY + ghW);
      ctx.lineTo(ghX - ghW, ghY + ghW * 0.5);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Glass polygonal walls (lit cyber violet)
      ctx.fillStyle = "rgba(167, 139, 250, 0.18)"; // Glass reflection
      ctx.strokeStyle = "rgba(16, 185, 129, 0.75)"; // Green glass beams
      ctx.lineWidth = 1.5;
      
      // Left glass panel
      ctx.beginPath();
      ctx.moveTo(ghX - ghW, ghY + ghW * 0.5);
      ctx.lineTo(ghX, ghY);
      ctx.lineTo(ghX, ghY - ghH);
      ctx.lineTo(ghX - ghW, ghY + ghW * 0.5 - ghH);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Right glass panel
      ctx.beginPath();
      ctx.moveTo(ghX, ghY + ghW * 0.5);
      ctx.lineTo(ghX + ghW, ghY + ghW * 0.5);
      ctx.lineTo(ghX + ghW, ghY + ghW * 0.5 - ghH);
      ctx.lineTo(ghX, ghY - ghH + ghW * 0.5);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Arch glass roof panels
      ctx.fillStyle = "rgba(147, 197, 253, 0.25)";
      ctx.beginPath();
      ctx.moveTo(ghX - ghW, ghY + ghW * 0.5 - ghH);
      ctx.lineTo(ghX, ghY - ghH - 12); // peak top of arch
      ctx.lineTo(ghX + ghW, ghY + ghW * 0.5 - ghH);
      ctx.lineTo(ghX, ghY - ghH);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Neon LED Internal Grow Lamp glow
      ctx.fillStyle = "rgba(236, 72, 153, 0.35)"; // Pink glow
      ctx.beginPath();
      ctx.arc(ghX, ghY - ghH, 8, 0, Math.PI * 2);
      ctx.fill();

      // Plant row silhouettes visible inside the greenhouse
      ctx.fillStyle = "#047857";
      ctx.beginPath();
      ctx.ellipse(ghX - 8, ghY + 8, 3, 5, Math.PI / 4, 0, Math.PI * 2);
      ctx.ellipse(ghX + 8, ghY + 12, 3, 5, -Math.PI / 4, 0, Math.PI * 2);
      ctx.fill();

      // --- 11. SHADED WATER TANK & SCAFFOLD BRACKETS ---
      // Placed on back-left coordinates
      const wTowerX = isoX - gridCount * tileW - 35;
      const wTowerY = isoY + gridCount * tileH - 70;

      // Sturdy metallic scaffolding pillars with criss-cross brace struts
      ctx.strokeStyle = "#475569"; // slate steel rails
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      
      const beamL = wTowerX - 22;
      const beamR = wTowerX + 22;
      const bBase = wTowerY + 105;

      // Left Pillar
      ctx.moveTo(wTowerX, wTowerY + 22);
      ctx.lineTo(beamL, bBase);

      // Right Pillar
      ctx.moveTo(wTowerX, wTowerY + 22);
      ctx.lineTo(beamR, bBase + 10);

      // Back support Pillar
      ctx.moveTo(wTowerX, wTowerY + 22);
      ctx.lineTo(wTowerX - 8, bBase - 15);
      ctx.stroke();

      // Horizontal steel braces
      ctx.strokeStyle = "#64748b";
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(wTowerX - 8, wTowerY + 45);
      ctx.lineTo(wTowerX + 8, wTowerY + 49);
      ctx.moveTo(wTowerX - 14, wTowerY + 72);
      ctx.lineTo(wTowerX + 14, wTowerY + 78);
      ctx.stroke();

      // X-Scaffolding bracing links
      ctx.beginPath();
      ctx.moveTo(wTowerX - 8, wTowerY + 45);
      ctx.lineTo(wTowerX + 14, wTowerY + 72);
      ctx.moveTo(wTowerX + 8, wTowerY + 49);
      ctx.lineTo(wTowerX - 14, wTowerY + 72);
      ctx.stroke();

      // Draw Main Cylindrical Water Reservoir (Real curved shading & glass volumetric gauge)
      const tankRadius = 20;
      const tankHeight = 32;

      // Outer tank steel capsule base
      ctx.fillStyle = "#cbd5e1";
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1.8;
      
      // Bottom ellipse shadow cap
      ctx.beginPath();
      ctx.ellipse(wTowerX, wTowerY + tankHeight, tankRadius, tankRadius * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Tank curved cylinder wall gradient
      const tankWallGrad = ctx.createLinearGradient(wTowerX - tankRadius, wTowerY, wTowerX + tankRadius, wTowerY);
      tankWallGrad.addColorStop(0, "#1e3a8a");   // Deep shadow indigo
      tankWallGrad.addColorStop(0.3, "#3b82f6"); // Reflective bright blue
      tankWallGrad.addColorStop(0.7, "#1d4ed8"); // Metallic blue sheen
      tankWallGrad.addColorStop(1, "#172554");   // Dark back curvature shadow
      
      ctx.fillStyle = tankWallGrad;
      ctx.beginPath();
      ctx.rect(wTowerX - tankRadius, wTowerY, tankRadius * 2, tankHeight);
      ctx.fill();
      ctx.stroke();

      // Glass telemetry slot strip on the front side showing live water movement
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.beginPath();
      ctx.rect(wTowerX - 3, wTowerY + 6, 6, tankHeight - 12);
      ctx.fill();

      // Volumetric fluid filling matching active values (pulsing water waving)
      const waveShift = Math.sin(tick * 0.1) * 3;
      ctx.fillStyle = "#38bdf8"; // cyan active fluid
      ctx.beginPath();
      ctx.rect(wTowerX - 3, wTowerY + 12 + waveShift, 6, tankHeight - 18 - waveShift);
      ctx.fill();

      // Round top roof cover dome (bio-shield shelter)
      const roofGrad = ctx.createRadialGradient(wTowerX, wTowerY, 2, wTowerX, wTowerY - 5, tankRadius);
      roofGrad.addColorStop(0, "#64748b");
      roofGrad.addColorStop(1, "#334155");
      ctx.fillStyle = roofGrad;
      ctx.beginPath();
      ctx.ellipse(wTowerX, wTowerY, tankRadius, tankRadius * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(wTowerX - tankRadius, wTowerY);
      ctx.quadraticCurveTo(wTowerX, wTowerY - 14, wTowerX + tankRadius, wTowerY);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Cyber labels
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 7px monospace";
      ctx.fillText(`H2O-R1`, wTowerX - 11, wTowerY + 22);

      // Active plumbing piping conduits extending from reservoir to fields
      ctx.strokeStyle = "rgba(29, 78, 216, 0.9)"; // High pressure blue pipes
      ctx.lineWidth = 3;
      const pipeStartX = wTowerX;
      const pipeStartY = wTowerY + tankHeight + 2;
      const pipeEndX = isoX - 60;
      const pipeEndY = isoY + 50;

      ctx.beginPath();
      ctx.moveTo(pipeStartX, pipeStartY);
      ctx.lineTo(pipeStartX + 10, pipeStartY + 10);
      ctx.lineTo(pipeEndX, pipeEndY);
      ctx.stroke();

      // Branches going layout to tilled paths
      for (let branch = 0; branch < 4; branch++) {
        ctx.strokeStyle = "rgba(14, 165, 233, 0.45)";
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        const startBX = pipeEndX + branch * 12;
        const startBY = pipeEndY + branch * 6;
        ctx.moveTo(startBX, startBY);
        ctx.lineTo(startBX + 105, startBY - 52);
        ctx.stroke();
      }

      // Animated Water Flow Particles through the primary high pressure line
      if (activeWaterValve) {
        ctx.fillStyle = "#38bdf8"; // bright neon blue droplet particles
        waterFlowParticles.forEach(part => {
          part.t += part.speed;
          if (part.t > 1) part.t = 0;
          
          const px = pipeStartX * (1 - part.t) + pipeEndX * part.t;
          const py = pipeStartY * (1 - part.t) + pipeEndY * part.t;
          
          ctx.beginPath();
          ctx.arc(px + 4, py + 4, 3.2, 0, Math.PI * 2);
          ctx.fill();

          // Sparky volumetric spray drops when hitting farm boundary
          if (tick % 24 < 12) {
            ctx.fillStyle = "rgba(56, 189, 248, 0.65)";
            ctx.beginPath();
            ctx.arc(pipeEndX + 8, pipeEndY + 12, Math.abs(Math.sin(tick * 0.2)) * 4.5, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }

      // --- 12. AUTONOMOUS FLYING DRONES (HIGH FIDELITY MULTI-SPECTRAL MAPPING) ---
      drones.forEach(drone => {
        drone.angleOffset += drone.speed;
        
        // Circular elliptical monitoring flight orbits
        const orbCenterX = isoX;
        const orbCenterY = isoY + 30;
        
        drone.x = orbCenterX + Math.cos(drone.angleOffset) * drone.radius;
        drone.y = orbCenterY - drone.hoverHeight + Math.sin(drone.angleOffset * 1.8) * (drone.radius * 0.3);

        // Volumetric Transparent Scanning lasers mapping fields
        const coneGrad = ctx.createLinearGradient(drone.x, drone.y, drone.x, drone.y + drone.hoverHeight + 40);
        coneGrad.addColorStop(0, "rgba(52, 211, 153, 0.48)");
        coneGrad.addColorStop(0.5, "rgba(52, 211, 153, 0.15)");
        coneGrad.addColorStop(1, "rgba(52, 211, 153, 0)");
        
        ctx.fillStyle = coneGrad;
        ctx.beginPath();
        ctx.moveTo(drone.x, drone.y + 4);
        ctx.lineTo(drone.x - 35, drone.y + drone.hoverHeight + 40);
        ctx.lineTo(drone.x + 35, drone.y + drone.hoverHeight + 40);
        ctx.closePath();
        ctx.fill();

        // Concentric scanning radar concentric target rings on crops
        ctx.strokeStyle = "rgba(16, 185, 129, 0.8)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.ellipse(drone.x, drone.y + drone.hoverHeight + 40, 18, 7, 0, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = "rgba(16, 185, 129, 0.3)";
        ctx.beginPath();
        ctx.ellipse(drone.x, drone.y + drone.hoverHeight + 40, 28, 11, 0, 0, Math.PI * 2);
        ctx.stroke();

        // Draw Drone Chassis (Polygonal Titanium Frame)
        ctx.fillStyle = "#1e293b"; // Charcoal black
        ctx.strokeStyle = "#475569";
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        // Drawing a sleek hexagonal fuselage shape
        ctx.moveTo(drone.x - 10, drone.y);
        ctx.lineTo(drone.x - 5, drone.y - 4);
        ctx.lineTo(drone.x + 5, drone.y - 4);
        ctx.lineTo(drone.x + 10, drone.y);
        ctx.lineTo(drone.x + 5, drone.y + 4);
        ctx.lineTo(drone.x - 5, drone.y + 4);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Flashing navigation LED (Wingtip green/red indicators)
        ctx.fillStyle = (tick % 24 < 12) ? "#10b981" : "#ef4444";
        ctx.beginPath();
        ctx.arc(drone.x - 10, drone.y, 1.5, 0, Math.PI * 2);
        ctx.arc(drone.x + 10, drone.y, 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Active Optical Camera Turret (Little rotating socket)
        const rotTurretOffset = Math.sin(tick * 0.05) * 3;
        ctx.fillStyle = drone.color; // Pulsing core optic green/blue
        ctx.beginPath();
        ctx.arc(drone.x + rotTurretOffset, drone.y + 4, 3, 0, Math.PI * 2);
        ctx.fill();

        // Quadrotor Truss Support Arms
        ctx.strokeStyle = "#64748b";
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        // Sturdy outward arms
        ctx.moveTo(drone.x - 10, drone.y);
        ctx.lineTo(drone.x - 24, drone.y - 3);
        ctx.moveTo(drone.x + 10, drone.y);
        ctx.lineTo(drone.x + 24, drone.y - 3);
        ctx.stroke();

        // Rotor Blades spinning animations (Circular blur polygons)
        const rSize = Math.sin(tick * 0.9) * 11;
        ctx.strokeStyle = "rgba(148, 163, 184, 0.9)";
        ctx.lineWidth = 1.2;
        
        ctx.beginPath();
        ctx.ellipse(drone.x - 24, drone.y - 4, Math.abs(rSize), 2, 0, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.ellipse(drone.x + 24, drone.y - 4, Math.abs(rSize), 2, 0, 0, Math.PI * 2);
        ctx.stroke();

        // Digital HUD overlay tag
        ctx.fillStyle = "#334155";
        ctx.font = "bold 8px monospace";
        ctx.fillText(drone.label, drone.x - 28, drone.y - 14);
        
        ctx.fillStyle = "#059669";
        ctx.font = "7px monospace";
        ctx.fillText(`AGRI-SPECTRAL SCAN LIVE`, drone.x - 30, drone.y - 7);
      });

      // --- 13. SPATIAL SIMULATOR HUD OVERLAY LABELS ---
      ctx.fillStyle = "#1e293b";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`AGRISENSE PRECISION FARM CAD v3.8`, 16, h - 35);
      
      ctx.fillStyle = "#475569";
      ctx.font = "8px monospace";
      ctx.fillText(`BREEZE FORCE: ${activeWeather.windSpeed} m/s // MODEL WEIGHT: XGBOOST ENGINE`, 16, h - 22);
      ctx.fillText(`FAO-56 METRIC INTEGRITY LEVEL: HIGH`, 16, h - 11);

      // Interactive flashing state button
      ctx.fillStyle = "rgba(16, 185, 129, 0.12)";
      ctx.fillRect(w - 140, h - 35, 124, 20);
      ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
      ctx.strokeRect(w - 140, h - 35, 124, 20);

      ctx.fillStyle = "#047857";
      ctx.font = "bold 8px monospace";
      ctx.fillText(`● HYDROMETRY STEWARD`, w - 134, h - 22);

      tick++;
      animFrame = requestAnimationFrame(resizeAndDraw);
    };

    resizeAndDraw();

    return () => {
      cancelAnimationFrame(animFrame);
    };
  }, [metrics, windIntensity, activeWaterValve, activeWeather]);

  const handleWindTweak = (level: number) => {
    setWindIntensity(level);
    const speed = level === 0.2 ? 1.1 : level === 1.0 ? 4.2 : 9.8;
    const desc = level === 0.2 ? "Calm Air" : level === 1.0 ? "Gentle Breeze" : "Gale Force Alert";
    setActiveWeather(prev => ({
      ...prev,
      windSpeed: speed,
      description: desc
    }));
  };

  return (
    <div className="space-y-8 animate-fade-in text-slate-800" id="natural-dashboard-root">
      
      {/* Dynamic Header Overview with Farmer Accents */}
      <div className="page-header-strip p-8">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="space-y-2 max-w-3xl">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 bg-amber-500 text-xs font-mono font-bold text-gray-900 rounded-lg">
              IOT CORES LIVE
            </span>
            <span className="text-xs text-emerald-100 font-semibold tracking-wide">
              North Block Sector-A Coordinate Logs
            </span>
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight sm:text-4xl text-white">
            AgriSense Command
          </h1>
          <p className="text-emerald-100/90 text-sm max-w-2xl font-medium">
            Merging ancient land stewardship with autonomous cyber-physical engines. Monitor multi-spectral health registries, control automated drip irrigation valves, and consult the AI model array below.
          </p>
        </div>

        <div className="flex flex-wrap gap-2.5 justify-start md:justify-end">
          <button 
            id="btn-shortcut-agrigpt"
            onClick={() => onNavigate("agrigpt")}
            className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-slate-900 font-bold text-xs rounded-xl shadow-md cursor-pointer flex items-center gap-1.5 transition-all active:scale-95"
          >
            <Activity className="w-4 h-4" />
            Talk AgriGPT AI
          </button>
          <button 
            onClick={() => onNavigate("twin")}
            className="px-4 py-2 bg-emerald-900/95 hover:bg-emerald-950 text-white border border-emerald-600 font-bold text-xs rounded-xl shadow-md cursor-pointer flex items-center gap-1.5 transition-all active:scale-95"
          >
            <Cpu className="w-4 h-4" />
            Digital ISO-Twin
          </button>
        </div>
        </div>
      </div>

      {/* Earthy Farmers Bento KPI Block Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        
        {/* Metric 1: Nitrogen Vigor */}
        <div className="p-5 rounded-2xl bg-white border border-stone-200 shadow-sm flex items-start justify-between relative overflow-hidden group hover:border-emerald-500 transition-all">
          <div className="space-y-1 z-10">
            <span className="text-[10px] font-bold text-stone-400 uppercase tracking-widest block font-mono">Soil Nitrogen [N]</span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-3xl font-extrabold text-stone-900 tracking-tight">{metrics.nitrogen} ppm</span>
            </div>
            <span className="text-[10.5px] text-emerald-600 block font-semibold flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-emerald-500 inline" /> Optimal organic feeding
            </span>
          </div>
          <div className="p-3 bg-stone-50 border border-stone-150 text-emerald-700 rounded-xl group-hover:bg-emerald-600 group-hover:text-white transition-all">
            <Sprout className="w-5 h-5" />
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 to-green-400" />
        </div>

        {/* Metric 2: Hydrology Profile */}
        <div className="p-5 rounded-2xl bg-white border border-stone-200 shadow-sm flex items-start justify-between relative overflow-hidden group hover:border-blue-500 transition-all">
          <div className="space-y-1 z-10">
            <span className="text-[10px] font-bold text-stone-400 uppercase tracking-widest block font-mono">Topsoil Moisture</span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-3xl font-extrabold text-stone-900 tracking-tight">{metrics.avgMoisture}%</span>
            </div>
            <span className={`text-[10.5px] block font-semibold ${metrics.avgMoisture < 35 ? 'text-amber-600' : 'text-blue-600'}`}>
              ◆ Water deficit level logged
            </span>
          </div>
          <div className="p-3 bg-stone-50 border border-stone-150 text-blue-600 rounded-xl group-hover:bg-blue-600 group-hover:text-white transition-all">
            <Droplet className="w-5 h-5" />
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-sky-400" />
        </div>

        {/* Metric 3: Weather Temp */}
        <div className="p-5 rounded-2xl bg-white border border-stone-200 shadow-sm flex items-start justify-between relative overflow-hidden group hover:border-amber-500 transition-all">
          <div className="space-y-1 z-10">
            <span className="text-[10px] font-bold text-stone-400 uppercase tracking-widest block font-mono">Ambient Temperature</span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-3xl font-extrabold text-stone-900 tracking-tight">{metrics.avgTemp}°C</span>
            </div>
            <span className="text-[10.5px] text-stone-500 block font-semibold">
              ● Transpiration index standard
            </span>
          </div>
          <div className="p-3 bg-stone-50 border border-stone-150 text-amber-600 rounded-xl group-hover:bg-amber-500 group-hover:text-white transition-all">
            <Thermometer className="w-5 h-5" />
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-amber-500 to-yellow-400" />
        </div>

        {/* Metric 4: AI Model Quality */}
        <div className="p-5 rounded-2xl bg-white border border-stone-200 shadow-sm flex items-start justify-between relative overflow-hidden group hover:border-purple-500 transition-all">
          <div className="space-y-1 z-10">
            <span className="text-[10px] font-bold text-stone-400 uppercase tracking-widest block font-mono">Model Array Health</span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-3xl font-extrabold text-stone-900 tracking-tight">{metrics.mlHealth}%</span>
            </div>
            <span className="text-[10.5px] text-purple-600 block font-semibold">
              ✔ 5 Edge Models Active
            </span>
          </div>
          <div className="p-3 bg-stone-50 border border-stone-150 text-purple-600 rounded-xl group-hover:bg-purple-600 group-hover:text-white transition-all">
            <Cpu className="w-5 h-5" />
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-purple-500 to-pink-400" />
        </div>

      </div>

      {/* Comprehensive Visual Farm Interactive Canvas Block */}
      <div className="bg-white rounded-2xl border border-stone-200 shadow-md overflow-hidden grid grid-cols-1 lg:grid-cols-12">
        
        {/* Left Span 8: The Animated 3D Farmland View */}
        <div className="lg:col-span-8 bg-gradient-to-b from-[#f9f5f0] to-[#f5ebd6] relative border-b lg:border-b-0 lg:border-r border-stone-200">
          
          {/* Floating UI Overlays */}
          <div className="absolute top-5 left-5 z-20 space-y-1 pointer-events-none">
            <span className="px-2 py-0.5 bg-slate-900/90 text-[10px] font-bold text-emerald-400 tracking-wider font-mono rounded-md">
              CYBERPHYSICAL MULTI-SPECTRAL EMULATOR
            </span>
            <h3 className="text-sm font-bold text-stone-800">Telemetry Spatial Twin Mapping</h3>
          </div>

          <div className="absolute top-5 right-5 z-20 flex flex-col gap-2">
            {/* Water Valve Override Button */}
            <button
              onClick={() => setActiveWaterValve(!activeWaterValve)}
              className={`p-1.5 px-3 rounded-lg text-[10px] font-bold font-mono transition-all border shadow cursor-pointer flex items-center gap-1 ${activeWaterValve ? 'bg-blue-600 text-white border-blue-500' : 'bg-white text-stone-600 border-stone-200'}`}
            >
              <Droplet className={`w-3 h-3 ${activeWaterValve ? 'animate-bounce' : ''}`} />
              Valve: {activeWaterValve ? 'Active Flowing' : 'Flow Suspended'}
            </button>
            
            {/* Wind controller tweak */}
            <div className="bg-white/80 backdrop-blur border border-stone-200 rounded-lg p-1 px-2 flex items-center gap-1.5 shadow">
              <span className="text-[9px] font-bold text-stone-500 uppercase font-mono">Breeze:</span>
              <button 
                onClick={() => handleWindTweak(0.2)}
                className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold ${windIntensity === 0.2 ? 'bg-amber-500 text-slate-950' : 'text-stone-600'}`}
              >
                Off
              </button>
              <button 
                onClick={() => handleWindTweak(1.0)}
                className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold ${windIntensity === 1.0 ? 'bg-amber-500 text-slate-950' : 'text-stone-600'}`}
              >
                Med
              </button>
              <button 
                onClick={() => handleWindTweak(2.4)}
                className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-semibold ${windIntensity === 2.4 ? 'bg-amber-500 text-slate-950' : 'text-stone-600'}`}
              >
                Storm
              </button>
            </div>
          </div>

          {/* Interactive HTML5 Canvas Container */}
          <div className="w-full relative">
            <canvas ref={mainCanvasRef} className="w-full block" />
          </div>

          {/* Simulation Legenda Margin */}
          <div className="p-3 px-5 bg-stone-100/50 border-t border-stone-200 text-xs font-mono text-stone-500 flex flex-wrap items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-[11px]">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 inline-block" /> Active high-growth Maize
            </span>
            <span className="flex items-center gap-1.5 text-[11px]">
              <span className="w-2.5 h-2.5 rounded bg-amber-500 inline-block" /> Native dry wheat crops
            </span>
            <span className="flex items-center gap-1.5 text-[11px]">
              <span className="w-2.5 h-2.5 rounded bg-blue-600 inline-block animate-pulse" /> Hydrological pipes line
            </span>
            <span className="flex items-center gap-1.5 text-[11px]">
              <span className="w-2 h-2 rounded-full bg-red-500 inline-block animate-ping" /> Real-time active IoT Node
            </span>
          </div>

        </div>

        {/* Right Span 4: The Farmers Manual Control Desk */}
        <div className="lg:col-span-4 p-6 space-y-6 flex flex-col justify-between bg-stone-50/60">
          
          <div className="space-y-4">
            <div className="pb-3 border-b border-stone-200/80">
              <h3 className="font-bold text-stone-900 text-sm tracking-tight flex items-center gap-2">
                <Wifi className="w-4 h-4 text-emerald-600" />
                Farmer Hardware Ingestion
              </h3>
              <p className="text-xs text-stone-500 mt-1">
                Receive, audit and model sensory packages transmitting from edge hardware probes over 2.4Ghz radio protocols.
              </p>
            </div>

            <div className="space-y-3">
              {/* Soil Composition Bars */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] font-mono text-stone-600">
                  <span>Nitrogen [N]</span>
                  <span className="font-bold text-stone-800">{metrics.nitrogen} ppm</span>
                </div>
                <div className="h-1.5 w-full bg-stone-200 rounded-full overflow-hidden">
                  <div className="bg-emerald-600 h-full rounded-full" style={{ width: `${Math.min(100, (metrics.nitrogen / 100) * 100)}%` }} />
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-[11px] font-mono text-stone-600">
                  <span>Phosphorus [P]</span>
                  <span className="font-bold text-stone-800">{metrics.phosphorus} ppm</span>
                </div>
                <div className="h-1.5 w-full bg-stone-200 rounded-full overflow-hidden">
                  <div className="bg-lime-600 h-full rounded-full" style={{ width: `${Math.min(100, (metrics.phosphorus / 80) * 100)}%` }} />
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-[11px] font-mono text-stone-600">
                  <span>Potassium [K]</span>
                  <span className="font-bold text-stone-800">{metrics.potassium} ppm</span>
                </div>
                <div className="h-1.5 w-full bg-stone-200 rounded-full overflow-hidden">
                  <div className="bg-amber-600 h-full rounded-full" style={{ width: `${Math.min(100, (metrics.potassium / 80) * 100)}%` }} />
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-[11px] font-mono text-stone-600">
                  <span>Soil acidity scale [pH]</span>
                  <span className="font-bold text-stone-800">{metrics.avgPH} pH</span>
                </div>
                <div className="h-1.5 w-full bg-stone-200 rounded-full overflow-hidden">
                  <div className="bg-teal-500 h-full rounded-full" style={{ width: `${(metrics.avgPH / 14) * 100}%` }} />
                </div>
              </div>
            </div>

            <div className="p-3.5 bg-stone-100 border border-stone-200 rounded-xl space-y-1 text-xs">
              <span className="font-mono text-[9px] text-stone-400 font-bold uppercase tracking-wider block">Connected Drones Command</span>
              <p className="text-stone-600 text-[11px] leading-relaxed">
                Sentinel-A and Sentinel-B hovering at orbits of **45m** and **60m** tracking multi-spectral vegetation indices continuously.
              </p>
            </div>
          </div>

          <div className="pt-2">
            <button
              onClick={() => onNavigate("sensors")}
              className="w-full py-2.5 bg-emerald-800 hover:bg-emerald-900 border border-emerald-950 font-bold text-xs tracking-wider uppercase text-white rounded-xl shadow-md transition-all active:scale-95 cursor-pointer"
            >
              Tune Hardware Coordinates
            </button>
          </div>

        </div>

      </div>

      {/* Main Modules & Short Alert Feeder Layout */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        
        {/* Left Side: Dynamic Farm Alert Logs (Span 2) */}
        <div className="lg:col-span-2 space-y-6">
          <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-sm space-y-6">
            <div className="flex items-center justify-between border-b border-stone-100 pb-3">
              <h2 className="text-base font-bold text-stone-900 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-500 animate-pulse" /> 
                Agronomic Alerts & Advisory
              </h2>
              <span className="px-2 py-0.5 bg-amber-50 border border-amber-200 text-amber-700 text-[10px] font-bold rounded-lg font-mono">
                2 ISSUES FLAGGED
              </span>
            </div>

            <div className="divide-y divide-stone-100">
              {latestAlerts.map((alert) => (
                <div key={alert.id} className="py-4 first:pt-0 last:pb-0 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div className="space-y-1 max-w-xl">
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${alert.severity === 'medium' ? 'bg-amber-500' : 'bg-emerald-500'}`} />
                      <h4 className="text-xs font-bold font-mono uppercase text-stone-800">{alert.title}</h4>
                    </div>
                    <p className="text-xs text-stone-500 leading-relaxed font-sans">{alert.desc}</p>
                  </div>
                  <button
                    id={`btn-advisory-${alert.id}`}
                    onClick={() => onNavigate(alert.target)}
                    className="px-4 py-1.5 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-xs font-bold text-emerald-800 transition-colors cursor-pointer border border-emerald-100 self-start sm:self-auto"
                  >
                    {alert.action}
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Core Modules Quick Short Link Cards */}
          <div className="space-y-4">
            <span className="text-[10px] font-bold text-stone-400 uppercase tracking-widest block font-mono">Primary Modules Access</span>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              
              <div 
                id="short-card-disease"
                onClick={() => onNavigate("disease")}
                className="p-5 rounded-2xl bg-stone-50 border border-stone-200 hover:border-emerald-500 cursor-pointer transition-all hover:-translate-y-0.5 space-y-3 group bg-white"
              >
                <div className="p-2.5 bg-emerald-50 text-emerald-700 rounded-xl w-fit border border-emerald-100 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                  <CloudSun className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-xs font-bold font-mono tracking-wide uppercase text-stone-900">Crop Health Vision</h4>
                  <p className="text-xs text-stone-500 mt-1 font-sans">Upload photos of leaves to detect fungal blights or mold spores.</p>
                </div>
              </div>

              <div 
                id="short-card-crop"
                onClick={() => onNavigate("crop")}
                className="p-5 rounded-2xl bg-stone-50 border border-stone-200 hover:border-emerald-500 cursor-pointer transition-all hover:-translate-y-0.5 space-y-3 group bg-white"
              >
                <div className="p-2.5 bg-emerald-50 text-emerald-700 rounded-xl w-fit border border-emerald-100 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                  <Sprout className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-xs font-bold font-mono tracking-wide uppercase text-stone-900">Crop Suitability</h4>
                  <p className="text-xs text-stone-500 mt-1 font-sans">Leverage ML random forests to select optimal seed classes for your soil pH.</p>
                </div>
              </div>

              <div 
                id="short-card-irrigation"
                onClick={() => onNavigate("irrigation")}
                className="p-5 rounded-2xl bg-stone-50 border border-stone-200 hover:border-emerald-500 cursor-pointer transition-all hover:-translate-y-0.5 space-y-3 group bg-white"
              >
                <div className="p-2.5 bg-emerald-50 text-emerald-700 rounded-xl w-fit border border-emerald-100 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                  <Droplet className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-xs font-bold font-mono tracking-wide uppercase text-stone-900">Irrigation Analytics</h4>
                  <p className="text-xs text-stone-500 mt-1 font-sans">Trigger dynamic soil flushing calculations based on live ET0 parameters.</p>
                </div>
              </div>

            </div>
          </div>

        </div>

        {/* Right Side: Farmers Almanac Soil & Climate Intel */}
        <div className="space-y-6">
          
          <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-sm space-y-6">
            <div className="border-b border-stone-100 pb-3">
              <h2 className="text-base font-bold text-stone-900 flex items-center gap-2">
                <CloudRain className="w-5 h-5 text-sky-500" /> Climatic Forecast
              </h2>
              <p className="text-xs text-stone-500 mt-0.5">North Sector Atmosphere</p>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs font-semibold p-2.5 bg-stone-50/50 border border-stone-150 rounded-xl">
                <span className="text-stone-500">Active Breeze speed:</span>
                <span className="font-mono text-stone-800 font-bold">{activeWeather.windSpeed} m/s</span>
              </div>

              <div className="flex items-center justify-between text-xs font-semibold p-2.5 bg-stone-50/50 border border-stone-150 rounded-xl">
                <span className="text-stone-500">Relative Humidity:</span>
                <span className="font-mono text-sky-600 font-bold">{activeWeather.humidity}% RH</span>
              </div>

              <div className="flex items-center justify-between text-xs font-semibold p-2.5 bg-stone-50/50 border border-stone-150 rounded-xl">
                <span className="text-stone-500">Solar radiation flux:</span>
                <span className="font-mono text-amber-600 font-bold">{activeWeather.solarRad} W/m²</span>
              </div>

              <div className="flex items-center justify-between text-xs font-semibold p-2.5 bg-stone-50/50 border border-stone-150 rounded-xl">
                <span className="text-stone-500">Atmosphere State:</span>
                <span className="text-emerald-700 font-bold">{activeWeather.description}</span>
              </div>
            </div>

            <div className="p-4 bg-emerald-50/50 border border-emerald-100 rounded-xl">
              <h4 className="text-xs font-bold text-emerald-800 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5" />
                FAO-56 Computational Model
              </h4>
              <p className="text-[10.5px] text-emerald-700 font-medium leading-relaxed mt-1.5">
                Reference evapotranspiration (ET0) is continuously evaluated against air temperature and wind velocities to adjust the irrigation schedules.
              </p>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
