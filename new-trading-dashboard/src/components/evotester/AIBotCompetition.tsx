import React, { useState } from 'react';

interface AIBotCompetitionProps {
  className?: string;
}

const AIBotCompetition: React.FC<AIBotCompetitionProps> = ({ className = '' }) => {
  const [isCompetitionActive, setIsCompetitionActive] = useState(false);
  const [loading, setLoading] = useState(false);

  return (
    <div className={`p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold">🤖 AI Bot Competition</h3>
          <p className="text-sm text-gray-600">Evolved strategies compete with micro-capital allocations</p>
        </div>
        <div className="flex space-x-2">
          {!isCompetitionActive ? (
            <button
              onClick={() => setIsCompetitionActive(true)}
              disabled={loading}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
            >
              ▶️ Start Competition
            </button>
          ) : (
            <>
              <button
                onClick={() => setIsCompetitionActive(false)}
                disabled={loading}
                className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 disabled:opacity-50"
              >
                🔄 Reallocate
              </button>
              <button
                onClick={() => setIsCompetitionActive(false)}
                disabled={loading}
                className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 disabled:opacity-50"
              >
                ⏹️ End Round
              </button>
            </>
          )}
        </div>
      </div>

      {/* Competition Status */}
      {isCompetitionActive && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-lg font-bold text-blue-600">3</div>
            <div className="text-xs text-gray-600">Active Bots</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-lg font-bold text-green-600">+4.3%</div>
            <div className="text-xs text-gray-600">Total Return</div>
          </div>
          <div className="text-center p-3 bg-purple-50 rounded-lg">
            <div className="text-lg font-bold text-purple-600">53</div>
            <div className="text-xs text-gray-600">Total Trades</div>
          </div>
          <div className="text-center p-3 bg-orange-50 rounded-lg">
            <div className="text-lg font-bold text-orange-600">5d</div>
            <div className="text-xs text-gray-600">Days Left</div>
          </div>
        </div>
      )}

      {/* Simple Leaderboard */}
      <div className="space-y-3">
        <h4 className="font-medium">🏆 Leaderboard</h4>

        <div className="border rounded-lg p-4 bg-green-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="bg-yellow-500 text-white px-2 py-1 rounded text-sm font-bold">#1</span>
              <div>
                <div className="font-medium">RSI-Momentum-V2</div>
                <div className="text-sm text-gray-600">SPY • Gen 47</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-green-600">+17.5%</div>
              <div className="text-sm text-gray-600">$587.50</div>
            </div>
          </div>
        </div>

        <div className="border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="bg-gray-500 text-white px-2 py-1 rounded text-sm font-bold">#2</span>
              <div>
                <div className="font-medium">Volatility-Breakout</div>
                <div className="text-sm text-gray-600">AAPL • Gen 32</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-blue-600">+6.3%</div>
              <div className="text-sm text-gray-600">$318.90</div>
            </div>
          </div>
        </div>

        <div className="border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="bg-orange-500 text-white px-2 py-1 rounded text-sm font-bold">#3</span>
              <div>
                <div className="font-medium">MeanReversion-Pro</div>
                <div className="text-sm text-gray-600">NVDA • Gen 28</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-red-600">-10.8%</div>
              <div className="text-sm text-gray-600">$178.40</div>
            </div>
          </div>
        </div>
      </div>

      {/* Competition Info */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium mb-2">🎯 Competition Rules</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Each bot starts with $100-$1,000 in capital</li>
          <li>• Winners get +20% more capital (snowball effect)</li>
          <li>• Losers get -50% less capital</li>
          <li>• Capital reallocates every hour automatically</li>
          <li>• Competition runs for 7 days</li>
        </ul>
      </div>
    </div>
  );
};

export default AIBotCompetition;
