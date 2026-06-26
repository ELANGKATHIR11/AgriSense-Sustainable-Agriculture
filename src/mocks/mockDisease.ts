/**
 * AGRISENSE Crop Disease & Pathology Mock Data
 */

import { DiseaseDetectionResult } from "../types";

export const mockDiseasePredictions: DiseaseDetectionResult[] = [
  {
    disease: "Tomato Leaf Mold (Passalora fulva)",
    confidence: 94.8,
    severity: "medium",
    symptoms: [
      "Pale green/yellow spot outlines forming upper leaf areas",
      "Olive-green velvet spores spreading extensively across underside regions",
      "Slight turgor loss with curling edge foliage structures"
    ],
    recommendations: [
      "Boost overhead air circulation inside high-tunnels immediately",
      "Transition watering lines to ground level sub-surface drip tapes to limit canopy moisture",
      "Conduct biological spray treatments using certified copper-octanoate or Bacillus subtilis ensembles"
    ]
  },
  {
    disease: "Powdery Mildew on Winter Squash",
    confidence: 88.5,
    severity: "low",
    symptoms: [
      "Talcum-like dusty white superficial mycelium patches on petioles and leaf veins",
      "Premature cellular necrosis and leaves brittle to touch",
      "Stunted vine elongation causing reduced solar absorption"
    ],
    recommendations: [
      "Coordinate mechanical pruning of dense shaded leaves to expose inner canopy to solar rays",
      "Perform foliar application using dilution spray of organic potassium bicarbonate or neem extract",
      "Maintain strict weeds exclusion boundaries to suppress alternate pathogen hosts"
    ]
  },
  {
    disease: "Redroot Pigweed (Amaranthus retroflexus)",
    confidence: 91.2,
    severity: "high",
    symptoms: [
      "Dense, aggressive weed stalks featuring reddish central taproots",
      "Fast-growing dense foliage directly intercepting crop root fertilizer reserves",
      "Clusters forming spikelet seeds in close proximity to primary irrigation tubes"
    ],
    recommendations: [
      "Enforce mechanical manual weed extraction prior to seed head maturation",
      "Lay organic straw or black UV-stabilized mulching film to limit vegetative sunlight penetration",
      "Apply selective early mechanical tilling on unplanted margins"
    ]
  },
  {
    disease: "Healthy Photosynthetic Crop Canopy",
    confidence: 97.9,
    severity: "low",
    symptoms: [
      "Uniform deep green chloroplast pigment distribution",
      "Robust turgor pressure within vascular plant cells",
      "No trace of pathogenic leaf curling, spot necrosis, or biting damage"
    ],
    recommendations: [
      "Persist with the LightGBM-optimized scheduled irrigation and moisture intervals",
      "Document baseline leaf spectral index within Agrisense registry",
      "Administer balanced compost tea for sustained microbiome vigor"
    ]
  }
];

export const getRandomDiseaseResult = (): DiseaseDetectionResult => {
  return mockDiseasePredictions[Math.floor(Math.random() * mockDiseasePredictions.length)];
};
