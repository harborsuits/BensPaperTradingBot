import { useUniverse } from "@/hooks/useUniverse";

export default function UniverseSwitcher() {
  const { list, watchlists, setUniverse } = useUniverse();
  if (!list.data || !watchlists.data) return null;
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground">Universe</span>
      <select
        className="border border-border rounded px-2 py-1 text-sm bg-background"
        value={list.data.id}
        onChange={(e) => setUniverse.mutate(e.target.value)}
      >
        {watchlists.data.map((w: any) => (
          <option key={w.id} value={w.id}>
            {w.id} ({w.count})
          </option>
        ))}
      </select>
    </div>
  );
}


