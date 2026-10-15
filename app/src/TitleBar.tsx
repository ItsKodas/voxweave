import type { ReactNode } from "react";

/** The window's own controls (the window has no system frame). Outside the app shell (tests) they do nothing. */
async function onWindow(action: "minimize" | "toggleMaximize" | "close") {
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow()[action]();
  } catch {
    // not running inside the app's window
  }
}

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="2" y="9" width="2.6" height="6" rx="1.3" fill="#6f7cff" />
      <rect x="6.6" y="5" width="2.6" height="14" rx="1.3" fill="#6f7cff" />
      <rect x="11.2" y="2" width="2.6" height="20" rx="1.3" fill="#8b95ff" />
      <rect x="15.8" y="6" width="2.6" height="12" rx="1.3" fill="#3dd6b0" />
      <rect x="20.4" y="9.5" width="2.6" height="5" rx="1.3" fill="#3dd6b0" />
    </svg>
  );
}

/**
 * The title bar: the app's name, whatever the screen puts in it (tabs, a status pill), and minimise, maximise and
 * close. Empty parts of the bar move the window; double-clicking them maximises it (data-tauri-drag-region).
 * Close does what the system close button did: the window hides to the tray and live mode keeps running.
 */
export function TitleBar({ children, status }: { children?: ReactNode; status?: ReactNode }) {
  return (
    <header className="titlebar" data-tauri-drag-region>
      <div className="brand" data-tauri-drag-region><Logo /> Voxweave</div>
      {children}
      <div className="drag" data-tauri-drag-region />
      {status}
      <div className="window-buttons">
        <button aria-label="Minimise" onClick={() => onWindow("minimize")}>
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><line x1="1.5" y1="6" x2="10.5" y2="6" stroke="currentColor" strokeWidth="1.3" /></svg>
        </button>
        <button aria-label="Maximise" onClick={() => onWindow("toggleMaximize")}>
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="1.5" y="1.5" width="9" height="9" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.3" /></svg>
        </button>
        <button className="close" aria-label="Close" onClick={() => onWindow("close")}>
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.3" /></svg>
        </button>
      </div>
    </header>
  );
}
