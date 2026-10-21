import React, { useState, useMemo } from 'react';
import {
  Search,
  Filter,
  Droplets,
  Calendar,
  TrendingUp,
  MapPin,
  Thermometer,
  Layers,
  ArrowUpDown,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  BookOpen
} from 'lucide-react';
import { CROPS_200_CATALOG, CropDirectoryItem } from '../data/cropsData';

interface CropDirectory200Props {
  onSelectCrop?: (crop: CropDirectoryItem) => void;
}

export const CropDirectory200: React.FC<CropDirectory200Props> = ({ onSelectCrop }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedSeason, setSelectedSeason] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'water' | 'duration' | 'yield'>('name');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState<number>(12);

  // Derive unique categories with item counts
  const categoriesWithCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    CROPS_200_CATALOG.forEach((crop) => {
      counts[crop.category] = (counts[crop.category] || 0) + 1;
    });
    return counts;
  }, []);

  const categories = useMemo(() => {
    return Object.keys(categoriesWithCounts).sort();
  }, [categoriesWithCounts]);

  // Unique seasons
  const seasons = useMemo(() => {
    const set = new Set<string>();
    CROPS_200_CATALOG.forEach((c) => set.add(c.season));
    return Array.from(set).sort();
  }, []);

  // Filter and sort crops
  const filteredCrops = useMemo(() => {
    return CROPS_200_CATALOG.filter((crop) => {
      const matchesCategory =
        selectedCategory === 'all' || crop.category.toLowerCase() === selectedCategory.toLowerCase();

      const matchesSeason =
        selectedSeason === 'all' || crop.season.toLowerCase().includes(selectedSeason.toLowerCase());

      const query = searchTerm.trim().toLowerCase();
      const matchesSearch =
        !query ||
        crop.name.toLowerCase().includes(query) ||
        crop.scientificName.toLowerCase().includes(query) ||
        crop.category.toLowerCase().includes(query) ||
        crop.season.toLowerCase().includes(query) ||
        crop.soilType.toLowerCase().includes(query) ||
        crop.majorStates.toLowerCase().includes(query);

      return matchesCategory && matchesSeason && matchesSearch;
    }).sort((a, b) => {
      let comparison = 0;
      if (sortBy === 'name') {
        comparison = a.name.localeCompare(b.name);
      } else if (sortBy === 'water') {
        comparison = a.waterRequirementMm - b.waterRequirementMm;
      } else if (sortBy === 'duration') {
        comparison = a.growthDurationDays - b.growthDurationDays;
      } else if (sortBy === 'yield') {
        comparison = a.expectedYieldTonHa - b.expectedYieldTonHa;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });
  }, [searchTerm, selectedCategory, selectedSeason, sortBy, sortOrder]);

  // Pagination calculations
  const totalPages = Math.ceil(filteredCrops.length / (pageSize === 0 ? filteredCrops.length || 1 : pageSize));
  const paginatedCrops = useMemo(() => {
    if (pageSize === 0) return filteredCrops;
    const start = (currentPage - 1) * pageSize;
    return filteredCrops.slice(start, start + pageSize);
  }, [filteredCrops, currentPage, pageSize]);

  const handleCategorySelect = (cat: string) => {
    setSelectedCategory(cat);
    setCurrentPage(1);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    setCurrentPage(1);
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Cereals & Millets':
        return 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20';
      case 'Pulses & Legumes':
        return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20';
      case 'Oilseeds':
        return 'bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/20';
      case 'Commercial & Cash Crops':
        return 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20';
      case 'Vegetables':
        return 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20';
      case 'Fruits & Plantation':
        return 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20';
      case 'Spices & Condiments':
        return 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20';
      case 'Floriculture':
        return 'bg-pink-500/10 text-pink-600 dark:text-pink-400 border-pink-500/20';
      default:
        return 'bg-teal-500/10 text-teal-600 dark:text-teal-400 border-teal-500/20';
    }
  };

  return (
    <div className="mt-8 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-2xl p-6 shadow-sm space-y-6">
      {/* Header and Live Search Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-[var(--agri-border)]">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-2 rounded-lg bg-[var(--agri-leaf)]/10 text-[var(--agri-leaf)]">
              <BookOpen className="w-5 h-5" />
            </span>
            <h2 className="text-xl font-bold text-[var(--agri-text)] tracking-tight">
              National Indian Crop Directory & Repository
            </h2>
          </div>
          <p className="text-xs text-[var(--agri-muted)] mt-1">
            Complete database of 200 agricultural crops verified by ICAR & TNAU agronomic standards.
          </p>
        </div>

        {/* Search Bar */}
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--agri-muted)]" />
          <input
            type="text"
            value={searchTerm}
            onChange={handleSearchChange}
            placeholder="Search crop, botanical name, state..."
            className="w-full pl-9 pr-4 py-2 text-xs bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-xl text-[var(--agri-text)] placeholder-[var(--agri-muted)] focus:outline-none focus:ring-1 focus:ring-[var(--agri-leaf)]"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-[var(--agri-muted)] hover:text-[var(--agri-text)]"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Category Filter Pills */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-[var(--agri-muted)]">
          <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5" /> Filter by Agronomic Category
          </span>
          <span className="font-mono-tabular">
            Showing {filteredCrops.length} of {CROPS_200_CATALOG.length} crops
          </span>
        </div>

        <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-thin">
          <button
            type="button"
            onClick={() => handleCategorySelect('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
              selectedCategory === 'all'
                ? 'bg-[#173D2B] text-white shadow-sm'
                : 'bg-[var(--agri-bg)] text-[var(--agri-muted)] hover:text-[var(--agri-text)] border border-[var(--agri-border)]'
            }`}
          >
            All Crops ({CROPS_200_CATALOG.length})
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() => handleCategorySelect(cat)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                selectedCategory === cat
                  ? 'bg-[#173D2B] text-white shadow-sm'
                  : 'bg-[var(--agri-bg)] text-[var(--agri-muted)] hover:text-[var(--agri-text)] border border-[var(--agri-border)]'
              }`}
            >
              {cat} ({categoriesWithCounts[cat] || 0})
            </button>
          ))}
        </div>
      </div>

      {/* Controls Row: Season Filter, Sort By, Items per page */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-[var(--agri-muted)]">Season:</span>
            <select
              value={selectedSeason}
              onChange={(e) => {
                setSelectedSeason(e.target.value);
                setCurrentPage(1);
              }}
              className="bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg px-2.5 py-1 text-[var(--agri-text)] focus:outline-none"
            >
              <option value="all">All Seasons</option>
              {seasons.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-[var(--agri-muted)] flex items-center gap-1">
              <ArrowUpDown className="w-3 h-3" /> Sort:
            </span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg px-2.5 py-1 text-[var(--agri-text)] focus:outline-none"
            >
              <option value="name">Crop Name</option>
              <option value="water">Water Need</option>
              <option value="duration">Duration</option>
              <option value="yield">Expected Yield</option>
            </select>

            <button
              type="button"
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="px-2 py-1 bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)] hover:bg-[var(--agri-border)]"
              title="Toggle sort direction"
            >
              {sortOrder === 'asc' ? '↑' : '↓'}
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[var(--agri-muted)]">Per page:</span>
          {[12, 24, 48, 0].map((size) => (
            <button
              key={size}
              onClick={() => {
                setPageSize(size);
                setCurrentPage(1);
              }}
              className={`px-2 py-1 rounded text-xs font-mono-tabular ${
                pageSize === size
                  ? 'bg-[var(--agri-leaf)] text-white font-bold'
                  : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
              }`}
            >
              {size === 0 ? 'All' : size}
            </button>
          ))}
        </div>
      </div>

      {/* 200 Crops Grid */}
      {paginatedCrops.length === 0 ? (
        <div className="py-16 text-center text-xs text-[var(--agri-muted)] bg-[var(--agri-bg)]/50 rounded-xl border border-dashed border-[var(--agri-border)]">
          No crops matched your search criteria "{searchTerm}". Try clearing search or selecting a different category.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {paginatedCrops.map((crop) => (
            <div
              key={crop.id}
              className="bg-[var(--agri-bg)] border border-[var(--agri-border)] hover:border-[var(--agri-leaf)] rounded-xl p-4 flex flex-col justify-between space-y-3 transition-all hover:shadow-md"
            >
              {/* Header: Name, Scientific Name, Badges */}
              <div>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="text-base font-bold text-[var(--agri-text)] leading-snug">
                      {crop.name}
                    </h3>
                    <p className="text-xs italic text-[var(--agri-muted)]">
                      {crop.scientificName}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-0.5 text-[10px] font-semibold rounded-md border whitespace-nowrap ${getCategoryColor(
                      crop.category
                    )}`}
                  >
                    {crop.category}
                  </span>
                </div>

                <div className="flex items-center gap-2 mt-2">
                  <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-[var(--agri-card)] text-[var(--agri-text)] border border-[var(--agri-border)]">
                    Season: <strong>{crop.season}</strong>
                  </span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-[var(--agri-card)] text-[var(--agri-muted)] border border-[var(--agri-border)]">
                    pH: {crop.optimalPh}
                  </span>
                </div>
              </div>

              {/* Core Agronomic Metrics Grid */}
              <div className="grid grid-cols-3 gap-2 py-2.5 px-3 rounded-lg bg-[var(--agri-card)]/70 border border-[var(--agri-border)]/60 text-xs font-mono-tabular">
                <div>
                  <span className="text-[10px] text-[var(--agri-muted)] block font-sans flex items-center gap-1">
                    <Droplets className="w-3 h-3 text-[var(--agri-water)]" /> Water
                  </span>
                  <span className="font-semibold text-[var(--agri-text)]">
                    {crop.waterRequirementMm} mm
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-[var(--agri-muted)] block font-sans flex items-center gap-1">
                    <Calendar className="w-3 h-3 text-[var(--agri-leaf)]" /> Duration
                  </span>
                  <span className="font-semibold text-[var(--agri-text)]">
                    {crop.growthDurationDays} d
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-[var(--agri-muted)] block font-sans flex items-center gap-1">
                    <TrendingUp className="w-3 h-3 text-[var(--agri-amber)]" /> Yield
                  </span>
                  <span className="font-semibold text-[var(--agri-text)]">
                    {crop.expectedYieldTonHa} t/ha
                  </span>
                </div>
              </div>

              {/* Additional Agronomic Details */}
              <div className="space-y-1.5 text-[11px] text-[var(--agri-muted)] pt-1">
                <div className="flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-[var(--agri-muted)] shrink-0" />
                  <span className="truncate">
                    Soil: <strong className="text-[var(--agri-text)]">{crop.soilType}</strong>
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Thermometer className="w-3.5 h-3.5 text-[var(--agri-muted)] shrink-0" />
                  <span>
                    Climate: {crop.temperatureRangeC} · Rain: {crop.rainfallRangeMm}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <MapPin className="w-3.5 h-3.5 text-[var(--agri-muted)] shrink-0" />
                  <span className="truncate">
                    States: <span className="text-[var(--agri-text)]">{crop.majorStates}</span>
                  </span>
                </div>
              </div>

              {/* Action Button: Load into recommendation engine */}
              {onSelectCrop && (
                <div className="pt-2 border-t border-[var(--agri-border)]">
                  <button
                    type="button"
                    onClick={() => onSelectCrop(crop)}
                    className="w-full py-1.5 px-3 text-xs font-semibold text-[var(--agri-leaf)] hover:text-white bg-[var(--agri-leaf)]/10 hover:bg-[#173D2B] rounded-lg transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    Load Parameters into AI Model
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pagination Controls */}
      {pageSize > 0 && totalPages > 1 && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-4 border-t border-[var(--agri-border)] text-xs text-[var(--agri-muted)]">
          <div>
            Showing {(currentPage - 1) * pageSize + 1} to{' '}
            {Math.min(currentPage * pageSize, filteredCrops.length)} of {filteredCrops.length} crops
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded-lg border border-[var(--agri-border)] bg-[var(--agri-bg)] disabled:opacity-40 cursor-pointer"
              title="Previous Page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum: number;
              if (totalPages <= 5) {
                pageNum = i + 1;
              } else if (currentPage <= 3) {
                pageNum = i + 1;
              } else if (currentPage >= totalPages - 2) {
                pageNum = totalPages - 4 + i;
              } else {
                pageNum = currentPage - 2 + i;
              }

              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  className={`w-8 h-8 rounded-lg text-xs font-mono-tabular transition-colors cursor-pointer ${
                    currentPage === pageNum
                      ? 'bg-[#173D2B] text-white font-bold'
                      : 'bg-[var(--agri-bg)] border border-[var(--agri-border)] text-[var(--agri-text)] hover:bg-[var(--agri-border)]'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}

            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded-lg border border-[var(--agri-border)] bg-[var(--agri-bg)] disabled:opacity-40 cursor-pointer"
              title="Next Page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
