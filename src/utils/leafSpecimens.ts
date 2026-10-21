export interface LeafSpecimenPreset {
  id: 'tomato_early_blight' | 'rice_bacterial_blight' | 'healthy_paddy';
  labelKey: 'sample1' | 'sample2' | 'sample3';
  cropHint: string;
  dataUrl: string;
}

function encodeSvgDataUrl(svg: string): string {
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg.trim())}`;
}

export const LEAF_SPECIMEN_PRESETS: LeafSpecimenPreset[] = [
  {
    id: 'tomato_early_blight',
    labelKey: 'sample1',
    cropHint: 'Tomato (Solanum lycopersicum)',
    dataUrl: encodeSvgDataUrl(`
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 280" width="400" height="280">
        <rect width="400" height="280" fill="#173D2B"/>
        <circle cx="340" cy="40" r="90" fill="#23533C" opacity="0.5"/>
        <!-- Leaf blade -->
        <path d="M60 220 C80 110, 180 50, 330 60 C310 170, 220 235, 60 220 Z" fill="#4A8C55" stroke="#2E6337" stroke-width="3"/>
        <!-- Midrib and veins -->
        <path d="M60 220 Q190 140 330 60" stroke="#8CC084" stroke-width="4" fill="none"/>
        <path d="M130 175 Q140 125 185 95" stroke="#8CC084" stroke-width="2" fill="none"/>
        <path d="M190 140 Q205 100 250 75" stroke="#8CC084" stroke-width="2" fill="none"/>
        <path d="M150 165 Q210 185 255 165" stroke="#8CC084" stroke-width="2" fill="none"/>
        <!-- Early blight concentric target lesions with chlorotic yellow halos -->
        <circle cx="170" cy="125" r="22" fill="#DCA849" opacity="0.85"/>
        <circle cx="170" cy="125" r="14" fill="#6E4726"/>
        <circle cx="170" cy="125" r="9" fill="none" stroke="#3D2310" stroke-width="2"/>
        <circle cx="170" cy="125" r="4" fill="#2B1709"/>

        <circle cx="235" cy="150" r="18" fill="#DCA849" opacity="0.85"/>
        <circle cx="235" cy="150" r="11" fill="#6E4726"/>
        <circle cx="235" cy="150" r="6" fill="none" stroke="#3D2310" stroke-width="1.8"/>

        <circle cx="265" cy="105" r="14" fill="#DCA849" opacity="0.8"/>
        <circle cx="265" cy="105" r="8" fill="#6E4726"/>
        <text x="18" y="262" fill="#DCE8D5" font-family="monospace" font-size="11">SPECIMEN: SOLANUM_LYCOPERSICUM // ALTERNARIA_SOLANI</text>
      </svg>
    `),
  },
  {
    id: 'rice_bacterial_blight',
    labelKey: 'sample2',
    cropHint: 'Paddy Rice (Oryza sativa)',
    dataUrl: encodeSvgDataUrl(`
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 280" width="400" height="280">
        <rect width="400" height="280" fill="#1B3B2B"/>
        <!-- Elongated paddy leaf -->
        <path d="M30 230 Q170 110 370 45 Q240 165 30 230 Z" fill="#4A8C55" stroke="#2D5E36" stroke-width="2.5"/>
        <!-- Marginal wavy yellowish-brown bacterial blight lesions from tip -->
        <path d="M210 105 Q290 70 370 45 Q310 105 235 125 Z" fill="#DCA849" opacity="0.9"/>
        <path d="M255 88 Q315 65 370 45 Q320 90 270 102 Z" fill="#9C6B30"/>
        <!-- Central midrib -->
        <path d="M30 230 Q200 135 370 45" stroke="#A7D49B" stroke-width="2.5" fill="none"/>
        <text x="18" y="262" fill="#DCE8D5" font-family="monospace" font-size="11">SPECIMEN: ORYZA_SATIVA // XANTHOMONAS_ORYZAE</text>
      </svg>
    `),
  },
  {
    id: 'healthy_paddy',
    labelKey: 'sample3',
    cropHint: 'healthy',
    dataUrl: encodeSvgDataUrl(`
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 280" width="400" height="280">
        <rect width="400" height="280" fill="#173D2B"/>
        <!-- Vibrant healthy leaf blade -->
        <path d="M55 225 C85 105, 190 45, 345 55 C315 170, 215 235, 55 225 Z" fill="#4A8C55" stroke="#68B074" stroke-width="2.5"/>
        <path d="M55 225 Q195 135 345 55" stroke="#B5E0B0" stroke-width="3.5" fill="none"/>
        <path d="M125 175 Q145 125 190 95" stroke="#8CC993" stroke-width="2" fill="none"/>
        <path d="M185 140 Q205 98 255 75" stroke="#8CC993" stroke-width="2" fill="none"/>
        <path d="M155 162 Q215 182 268 158" stroke="#8CC993" stroke-width="2" fill="none"/>
        <!-- Dew drops -->
        <circle cx="165" cy="135" r="5" fill="#E3F4FC" opacity="0.85"/>
        <circle cx="240" cy="115" r="4" fill="#E3F4FC" opacity="0.85"/>
        <text x="18" y="262" fill="#DCE8D5" font-family="monospace" font-size="11">SPECIMEN: ORYZA_SATIVA // HEALTHY_CONTROL_SPAD_41</text>
      </svg>
    `),
  },
];
