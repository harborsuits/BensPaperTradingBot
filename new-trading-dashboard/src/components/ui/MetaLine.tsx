import React from 'react';

interface MetaLineProps {
  meta?: {
    asOf?: string;
    source?: string;
    latency?: number;
    version?: string;
    [key: string]: any;
  };
  className?: string;
}

/**
 * A small component to display metadata information in a consistent format
 * Typically used at the bottom of cards to show data freshness and source
 */
const MetaLine: React.FC<MetaLineProps> = ({ meta = {}, className = '' }) => {
  // Format timestamp
  const formatTime = (timestamp?: string) => {
    if (!timestamp) return '—';
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return '—';
    }
  };

  // Format latency
  const formatLatency = (latency?: number) => {
    if (latency === undefined) return null;
    if (latency < 1) return '<1ms';
    if (latency > 1000) return `${(latency / 1000).toFixed(1)}s`;
    return `${Math.round(latency)}ms`;
  };

  return (
    <div className={`text-xs text-muted-foreground flex flex-wrap gap-x-2 ${className}`}>
      {meta.asOf && (
        <span>
          As of {formatTime(meta.asOf)}
        </span>
      )}
      
      {meta.source && (
        <>
          <span className="text-muted-foreground">•</span>
          <span>
            Source: {meta.source}
          </span>
        </>
      )}
      
      {meta.latency !== undefined && (
        <>
          <span className="text-muted-foreground">•</span>
          <span>
            Latency: {formatLatency(meta.latency)}
          </span>
        </>
      )}
      
      {meta.version && (
        <>
          <span className="text-muted-foreground">•</span>
          <span>
            v{meta.version}
          </span>
        </>
      )}
      
      {/* Render any additional metadata fields */}
      {Object.entries(meta)
        .filter(([key]) => !['asOf', 'source', 'latency', 'version'].includes(key))
        .map(([key, value]) => (
          <React.Fragment key={key}>
            <span className="text-muted-foreground">•</span>
            <span>
              {key}: {String(value)}
            </span>
          </React.Fragment>
        ))}
    </div>
  );
};

export default MetaLine;