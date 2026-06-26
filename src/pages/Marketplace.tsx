import React, { useState, useEffect } from "react";
import { ShoppingCart, Search, Filter, Tag, Check, ExternalLink, Sparkles, Sprout } from "lucide-react";

interface Product {
  id: number;
  name: string;
  category: string;
  price: number;
  supplier: string;
  buy_url: string;
  description: string;
}

export default function Marketplace() {
  const [products, setProducts] = useState<Product[]>([]);
  const [recommendations, setRecommendations] = useState<Product[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedDisease, setSelectedDisease] = useState("Tomato Leaf Mold");
  const [cart, setCart] = useState<Product[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchProducts = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/marketplace/products");
      const data = await res.json();
      setProducts(data);
    } catch (e) {
      console.error("Failed to load products", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchRecommendations = async () => {
    try {
      const res = await fetch(`/api/marketplace/recommendations?disease=${encodeURIComponent(selectedDisease)}`);
      const data = await res.json();
      setRecommendations(data);
    } catch (e) {
      console.error("Failed to load recommendations", e);
    }
  };

  useEffect(() => {
    fetchProducts();
  }, []);

  useEffect(() => {
    fetchRecommendations();
  }, [selectedDisease]);

  const handleAddToCart = (product: Product) => {
    if (!cart.some((p) => p.id === product.id)) {
      setCart([...cart, product]);
    }
  };

  const filteredProducts = products.filter((product) => {
    const matchesSearch = product.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          product.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === "all" || product.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="page-header-strip px-6 py-6 md:py-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 text-white">
        <div>
          <span className="agri-badge bg-amber-500/20 border-amber-400 text-amber-300 mb-2">Edge Commerce Hub</span>
          <h1 className="text-2xl md:text-3xl font-black tracking-tight font-sans">Agri Marketplace</h1>
          <p className="text-xs text-emerald-100/70 mt-1 font-mono">Precision supplies, fertilizers, seeds, and smart diagnostics recommendation.</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative bg-black/20 border border-white/10 px-4 py-2.5 rounded-xl flex items-center gap-2">
            <ShoppingCart className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-bold font-mono">Cart ({cart.length})</span>
          </div>
        </div>
      </div>

      {/* AI Recommendation Drawer */}
      <div className="soil-panel p-5 text-white">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center border border-amber-500/30">
            <Sparkles className="w-4 h-4 text-amber-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold tracking-tight">AI Treatment Assistant</h3>
            <p className="text-[10px] text-emerald-400/60 font-mono">Recommending fertilizers & inputs based on detected crop disease profiles.</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
          <div className="bg-black/20 border border-white/5 p-4 rounded-xl space-y-3">
            <label className="block text-[10px] font-bold uppercase tracking-wider text-emerald-400 font-mono">
              Target Crop Anomaly
            </label>
            <select
              value={selectedDisease}
              onChange={(e) => setSelectedDisease(e.target.value)}
              className="w-full px-3 py-2 bg-emerald-950 border border-emerald-800 rounded-lg text-xs font-medium text-white outline-none focus:border-amber-400 transition-colors"
            >
              <option value="Tomato Leaf Mold">Tomato Leaf Mold (Fungal)</option>
              <option value="Maize Nitrogen Deficit">Maize Soil Deficit (Nutrient)</option>
              <option value="Bacterial Blight">Bacterial Leaf Blight (Bacterial)</option>
            </select>
            <p className="text-[10px] text-emerald-100/50 leading-relaxed">
              Florence-2 or TabPFN diagnostic inference logs map treatment inputs to local distributors.
            </p>
          </div>

          <div className="md:col-span-2 space-y-2">
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-amber-400 font-mono">Recommended Inputs</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {recommendations.length > 0 ? (
                recommendations.map((prod) => (
                  <div key={`rec-${prod.id}`} className="bg-white/[0.04] border border-white/10 rounded-xl p-3 flex flex-col justify-between hover:bg-white/[0.08] transition-colors">
                    <div>
                      <span className="text-[8px] font-bold font-mono text-amber-400 uppercase tracking-wider px-1.5 py-0.5 bg-amber-500/10 border border-amber-400/20 rounded">AI Target Match</span>
                      <h5 className="text-xs font-bold mt-1 text-white">{prod.name}</h5>
                      <p className="text-[10px] text-emerald-100/60 mt-1 line-clamp-2 leading-snug">{prod.description}</p>
                    </div>
                    <div className="flex items-center justify-between mt-3 pt-2.5 border-t border-white/5">
                      <span className="text-xs font-bold text-amber-300 font-mono">${prod.price.toFixed(2)}</span>
                      <button
                        onClick={() => handleAddToCart(prod)}
                        className="text-[10px] font-bold px-2.5 py-1 bg-amber-500 text-amber-950 hover:bg-amber-400 transition-all rounded font-sans cursor-pointer"
                      >
                        Add to Cart
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-xs text-white/40">No recommendations found for this anomaly type.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Catalog View */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Filters Sidebar */}
        <div className="agri-card space-y-5 h-fit lg:sticky lg:top-4">
          <div>
            <h3 className="text-xs font-bold text-[#0f2e1e] uppercase tracking-wider mb-2 font-mono flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5" /> Search Catalog
            </h3>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Fungicide, seeds, etc..."
              className="agri-input font-sans text-xs"
            />
          </div>

          <div>
            <h3 className="text-xs font-bold text-[#0f2e1e] uppercase tracking-wider mb-2 font-mono flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5" /> Category Filter
            </h3>
            <div className="flex flex-col gap-1.5">
              {[
                { id: "all", label: "All Categories" },
                { id: "seed", label: "Crop Seeds" },
                { id: "fertilizer", label: "Soil Boosters" },
                { id: "pesticide", label: "Crop Health / Pesticides" }
              ].map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={`text-left px-3 py-2 text-xs font-medium rounded-lg transition-colors cursor-pointer ${
                    selectedCategory === cat.id
                      ? "bg-emerald-950 text-white"
                      : "text-emerald-800/80 hover:bg-emerald-50"
                  }`}
                >
                  {cat.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Product Cards Grid */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-emerald-800/60 font-mono">
              Showing {filteredProducts.length} precision products
            </span>
          </div>

          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[1, 2, 3].map((n) => (
                <div key={n} className="agri-card h-64 skeleton" />
              ))}
            </div>
          ) : filteredProducts.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {filteredProducts.map((prod) => (
                <div key={prod.id} className="agri-card flex flex-col justify-between hover:shadow-xl transition-all">
                  <div>
                    <span className="agri-chip chip-gray mb-2.5">{prod.category}</span>
                    <h4 className="text-sm font-bold text-[#0f2e1e] tracking-tight line-clamp-1">{prod.name}</h4>
                    <p className="text-[10px] text-emerald-900/60 font-mono mt-0.5">Supplier: {prod.supplier}</p>
                    <p className="text-xs text-emerald-800/80 mt-2 line-clamp-3 leading-relaxed">{prod.description}</p>
                  </div>
                  <div className="flex items-center justify-between border-t border-emerald-900/5 mt-4 pt-3">
                    <span className="text-sm font-black text-emerald-950 font-mono">${prod.price.toFixed(2)}</span>
                    <button
                      onClick={() => handleAddToCart(prod)}
                      className="btn-primary py-1 px-3 text-[11px] font-sans tracking-tight cursor-pointer"
                    >
                      Buy Now
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <Sprout className="w-12 h-12 text-emerald-400" />
              <p className="text-sm text-emerald-800 font-bold">No products match your filters.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
