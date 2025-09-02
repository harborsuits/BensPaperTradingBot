import React from 'react';

type DropdownContextType = { open: boolean; setOpen: (v: boolean) => void } | null;
const DropdownContext = React.createContext<DropdownContextType>(null);

type DropdownMenuProps = { children: React.ReactNode };

export const DropdownMenu: React.FC<DropdownMenuProps> = ({ children }) => {
  const [open, setOpen] = React.useState(false);
  return (
    <DropdownContext.Provider value={{ open, setOpen }}>
      <div className="relative inline-block text-left">{children}</div>
    </DropdownContext.Provider>
  );
};

type TriggerProps = { asChild?: boolean; children: React.ReactNode } & React.ComponentProps<'button'>;
export const DropdownMenuTrigger: React.FC<TriggerProps> = ({ children, className = '', ...props }) => {
  const ctx = React.useContext(DropdownContext)!;
  return (
    <button
      type="button"
      className={`inline-flex items-center gap-1 px-2 py-1 rounded border ${className}`}
      onClick={() => ctx.setOpen(!ctx.open)}
      {...props}
    >
      {children}
    </button>
  );
};

type ContentProps = { className?: string; children: React.ReactNode };
export const DropdownMenuContent: React.FC<ContentProps> = ({ className, children }) => {
  const ctx = React.useContext(DropdownContext)!;
  if (!ctx.open) return null;
  return (
    <div className={`absolute right-0 z-50 mt-2 min-w-[10rem] rounded-md border border-border bg-card p-1 shadow ${className || ''}`}>
      {children}
    </div>
  );
};

export const DropdownMenuLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">{children}</div>
);

export const DropdownMenuSeparator: React.FC = () => <div className="my-1 h-px bg-border" />;

type CheckboxItemProps = {
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  children: React.ReactNode;
};
export const DropdownMenuCheckboxItem: React.FC<CheckboxItemProps> = ({ checked = false, onCheckedChange, children }) => (
  <button
    type="button"
    onClick={() => onCheckedChange && onCheckedChange(!checked)}
    className="flex w-full items-center px-2 py-1.5 text-left text-sm hover:bg-muted"
  >
    <input type="checkbox" className="mr-2" checked={checked} readOnly />
    {children}
  </button>
);

export default DropdownMenu;

