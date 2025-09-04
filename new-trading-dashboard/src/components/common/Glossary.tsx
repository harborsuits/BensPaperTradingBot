import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip';

const GLOSSARY: Record<string,string> = {
  Long: 'A bet that the price will go up.',
  Short: 'A bet that the price will go down.',
  Spread: 'The gap between bid and ask prices; wider spreads increase trading cost.',
  IV: 'Implied volatility: the market’s expectation of future price swings.',
  MA: 'Moving Average: average price over a period (e.g., 20-day).',
};

export function Term({ children }:{ children: keyof typeof GLOSSARY }){
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="underline decoration-dotted cursor-help text-[12px]">{children}</span>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-xs">{GLOSSARY[children]}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
