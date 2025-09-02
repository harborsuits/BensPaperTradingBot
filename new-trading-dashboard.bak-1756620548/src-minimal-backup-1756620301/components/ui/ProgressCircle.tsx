import React from 'react';
import styles from './ProgressCircle.module.css';

// Helper function to get CSS color class based on color string
const getColorClass = (color: string = '#3182CE'): string => {
  const colorMap: Record<string, string> = {
    '#3182CE': styles.colorBlue,
    '#48BB78': styles.colorGreen,
    '#ED8936': styles.colorOrange,
    '#E53E3E': styles.colorRed,
    '#805AD5': styles.colorPurple,
    '#38B2AC': styles.colorTeal,
    '#D53F8C': styles.colorPink,
    '#A0AEC0': styles.colorGray,
  };
  
  return colorMap[color] || styles.colorBlue; // Default to blue if no match
};

interface ProgressCircleProps {
  progress: number; // Value between 0 and 1
  size?: number;
  strokeWidth?: number;
  circleColor?: string;
  progressColor?: string;
  showValue?: boolean;
  className?: string;
}

export const ProgressCircle: React.FC<ProgressCircleProps> = ({
  progress,
  size = 40,
  strokeWidth = 4,
  circleColor = '#E2E8F0',
  progressColor = '#3182CE',
  showValue = true,
  className = '',
}) => {
  // Ensure progress is between 0 and 1
  const normalizedProgress = Math.min(Math.max(progress, 0), 1);
  
  // Calculate circle properties
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - normalizedProgress);
  
  // Center of the circle
  const center = size / 2;
  
  return (
    <div className={`inline-flex items-center justify-center ${className}`}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={circleColor}
          strokeWidth={strokeWidth}
        />
        
        {/* Progress circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={progressColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-500 ease-in-out"
        />
      </svg>
      
      {showValue && (
        <div 
          className={`${styles.progressText} ${getColorClass(progressColor)}`}
        >
          {Math.round(normalizedProgress * 100)}%
        </div>
      )}
    </div>
  );
};

export default ProgressCircle;
