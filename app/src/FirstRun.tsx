import { useCallback, useEffect, useRef, useState } from "react";
import { rememberMode, setupApi, SetupState } from "./setupApi";

/** What the setup downloads: Python 22 MB, the packages 124 MB and the base speech model 378 MB, measured on
 * the first real run (README, Install). */
export const DOWNLOAD_SIZE = "about 525 MB";
export const DISK_SIZE = "about 0.9 GB";

/**
 * The shell's setup state, followed through its events. Null until the shell has answered, and for good when
 * there is no shell to ask; the app then behaves as the source checkout does.
 */
export function useSetup(): { setup: SetupState | null; retry: () => void } {
  const [setup, setSetup] = useState<SetupState | null>(null);
  const events = useRef(0); // an event that arrives while a question is open is newer than its answer

  const take = useCallback((state: SetupState) => {
    rememberMode(state);
    setSetup(state);
  }, []);

  useEffect(() => {
    let gone = false;
    const listening = setupApi
      .onProgress((state) => {
        events.current += 1;
        if (!gone) take(state);
      })
      .catch(() => null);
    const before = events.current;
    setupApi.state().then(
      (state) => {
        if (!gone && events.current === before) take(state);
      },
      () => {},
    );
    return () => {
      gone = true;
      void listening.then((off) => off?.());
    };
  }, [take]);

  const retry = useCallback(() => {
    const before = events.current;
    setupApi.start().then(
      (state) => {
        if (events.current === before) take(state);
      },
      () => {},
    );
  }, [take]);

  return { setup, retry };
}

/** Shown by the installed app until its one-time setup has finished. */
export function FirstRun({ setup, onRetry }: { setup: SetupState; onRetry: () => void }) {
  const failed = setup.status === "failed";
  return (
    <section className="first-run">
      <h2>First-time setup</h2>
      <p>
        Voxweave is downloading what its voice engine needs: Python, the engine's packages and the base speech
        model, {DOWNLOAD_SIZE} in total ({DISK_SIZE} on disk).
      </p>
      <p>It needs an internet connection this once. After that, live mode works without one.</p>
      <div className="train-bar">
        <progress max={100} value={setup.percent} aria-label="Setup progress" />
        <span>{Math.round(setup.percent)}%</span>
      </div>
      {!failed && <p className="progress-line">{setup.message}</p>}
      {failed && (
        <>
          <div role="alert" className="error">
            <p>The setup did not finish.</p>
            <p>{setup.error ?? "No reason was given."}</p>
            <p>The details are in {setup.setup_log}</p>
          </div>
          <div className="actions">
            <button onClick={onRetry}>Retry</button>
          </div>
          <p className="hint">Retry continues where the setup stopped; nothing is downloaded twice.</p>
        </>
      )}
      <p className="progress-line">Everything is stored in {setup.root}</p>
    </section>
  );
}
