import { getCurrentWebview } from "@tauri-apps/api/webview";

/** Calls handler with the file paths whenever files are dropped on the app window. */
export async function onFilesDropped(handler: (paths: string[]) => void): Promise<() => void> {
  return getCurrentWebview().onDragDropEvent((event) => {
    if (event.payload.type === "drop" && event.payload.paths.length > 0) handler(event.payload.paths);
  });
}
