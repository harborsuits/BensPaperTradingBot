import { useState } from "react";
import { useUniverse } from "@/hooks/useUniverse";
import { ChevronDown, Globe, Zap, TrendingUp, Search } from "lucide-react";

const watchlistIcons = {
  default: Globe,
  small_caps_liquid: Zap,
  etfs_top: TrendingUp,
  news_movers_today: Search,
};

export default function UniverseSwitcher() {
  const { watchlists, setUniverse } = useUniverse();
  const [isOpen, setIsOpen] = useState(false);
  const [currentUniverse, setCurrentUniverse] = useState("small_caps_liquid");

  const handleSwitch = async (watchlistId: string) => {
    try {
      await setUniverse.mutateAsync(watchlistId);
      setCurrentUniverse(watchlistId);
      setIsOpen(false);
    } catch (error) {
      console.error("Failed to switch universe:", error);
    }
  };

  const currentWatchlist = watchlists.data?.find(w => w.id === currentUniverse) ||
                          watchlists.data?.find(w => w.id === "small_caps_liquid") ||
                          watchlists.data?.[0];

  const IconComponent = currentWatchlist ? watchlistIcons[currentWatchlist.id as keyof typeof watchlistIcons] || Globe : Globe;

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs rounded px-3 py-2 bg-blue-800/60 border border-blue-600 text-white hover:bg-blue-700/60 transition-colors"
      >
        <IconComponent size={14} />
        <span className="hidden sm:inline">
          {currentWatchlist?.name || "Loading..."} ({currentWatchlist?.symbols?.length || 0})
        </span>
        <span className="sm:hidden">
          {currentWatchlist?.symbols?.length || 0}
        </span>
        <ChevronDown size={12} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 py-2 bg-slate-900 border border-slate-700 rounded-md shadow-lg z-50">
          <div className="px-3 py-2 text-xs font-medium text-slate-400 border-b border-slate-700">
            Symbol Universes
          </div>
          {watchlists.data?.map((watchlist) => {
            const Icon = watchlistIcons[watchlist.id as keyof typeof watchlistIcons] || Globe;
            const isActive = watchlist.id === currentUniverse;

            return (
              <button
                key={watchlist.id}
                onClick={() => handleSwitch(watchlist.id)}
                className={`w-full text-left px-3 py-2 text-xs hover:bg-slate-800 flex items-center gap-2 ${
                  isActive ? 'bg-blue-800/30 text-blue-300' : 'text-slate-300'
                }`}
              >
                <Icon size={14} />
                <div className="flex-1">
                  <div className="font-medium">{watchlist.name}</div>
                  <div className="text-slate-500">{watchlist.symbols?.length || 0} symbols</div>
                </div>
                {isActive && <div className="w-2 h-2 bg-blue-500 rounded-full"></div>}
              </button>
            );
          })}
          <div className="px-3 py-2 text-xs text-slate-500 border-t border-slate-700 mt-2">
            <Zap size={12} className="inline mr-1" />
            Open discovery active
          </div>
        </div>
      )}
    </div>
  );
}


