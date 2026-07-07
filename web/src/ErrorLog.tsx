import { useErrors } from './api';
import { timeAgo } from './format';

export default function ErrorLog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const errors = useErrors();

  if (!open) return null;

  return (
    <div className="fixed bottom-4 right-20 z-40 w-80 sm:w-96">
      <div className="card max-h-80 overflow-hidden">
        <header className="flex items-center justify-between border-b border-line bg-surface-2/50 px-3 py-2 dark:bg-neutral-800/40">
          <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500">Recent errors</span>
          <button onClick={onClose} className="text-2xs text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300">
            ✕
          </button>
        </header>
        <div className="max-h-64 overflow-y-auto p-2">
          {errors.length === 0 ? (
            <div className="px-2 py-4 text-center text-2xs text-neutral-500">No recent errors</div>
          ) : (
            <ul className="space-y-2">
              {errors.map(e => (
                <li key={e.id} className="rounded-md border border-danger/20 bg-danger-dim/30 p-2 text-xs text-danger">
                  <div className="flex items-center gap-1.5">
                    <span className="rounded bg-danger/20 px-1 py-0.5 text-2xs font-medium uppercase">{e.source}</span>
                    <span className="text-2xs text-neutral-500">{timeAgo(e.ts)}</span>
                  </div>
                  <div className="mt-1 break-words">{e.message}</div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
