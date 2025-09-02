import React, { useState, useRef, useEffect } from 'react';

type TooltipCtx = {
  open: boolean;
  setOpen: (v: boolean) => void;
  triggerRef: React.RefObject<HTMLDivElement>;
};

const Ctx = React.createContext<TooltipCtx | null>(null);

export function TooltipProvider({ children }: { children: React.ReactNode }) {
  // Minimal provider for API compatibility
  return <>{children}</>;
}

export function Tooltip({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLDivElement>(null);
  return (
    <Ctx.Provider value={{ open, setOpen, triggerRef }}>
      <div className="relative inline-block">{children}</div>
    </Ctx.Provider>
  );
}

export function TooltipTrigger({ asChild, children }: { asChild?: boolean; children: React.ReactElement }) {
  const ctx = React.useContext(Ctx)!;
  const handlers = {
    onMouseEnter: () => ctx.setOpen(true),
    onMouseLeave: () => ctx.setOpen(false),
    onFocus: () => ctx.setOpen(true),
    onBlur: () => ctx.setOpen(false),
  } as const;
  if (asChild) {
    return React.cloneElement(children, { ref: ctx.triggerRef, ...handlers });
  }
  return (
    <div ref={ctx.triggerRef} {...handlers}>
      {children}
    </div>
  );
}

export function TooltipContent({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  const ctx = React.useContext(Ctx)!;
  if (!ctx.open) return null;
  return (
    <div className={`absolute left-1/2 z-50 mt-2 -translate-x-1/2 whitespace-nowrap rounded-md bg-gray-800 px-3 py-2 text-xs text-white shadow ${className}`} role="tooltip">
      {children}
    </div>
  );
}

// Back-compat default export: a simplified single-element tooltip
export default function DefaultTooltip({ content, children }: { content: React.ReactNode; children: React.ReactElement }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative inline-block">
      {React.cloneElement(children, {
        onMouseEnter: () => setOpen(true),
        onMouseLeave: () => setOpen(false),
        onFocus: () => setOpen(true),
        onBlur: () => setOpen(false),
      })}
      {open && (
        <div className="absolute left-1/2 z-50 mt-2 -translate-x-1/2 whitespace-nowrap rounded-md bg-gray-800 px-3 py-2 text-xs text-white shadow">
          {content}
        </div>
      )}
    </div>
  );
}
