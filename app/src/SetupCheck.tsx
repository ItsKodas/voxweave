import { useEffect, useState } from "react";
import { api, Check } from "./api";

/** Lists failing setup checks. Changing `refreshKey` runs the checks again. */
export function SetupCheck({ refreshKey = 0 }: { refreshKey?: number }) {
  const [failing, setFailing] = useState<Check[]>([]);
  useEffect(() => {
    api.setupCheck().then((checks) => setFailing(checks.filter((c) => !c.ok))).catch(() => setFailing([]));
  }, [refreshKey]);
  if (failing.length === 0) return null;
  return (
    <section className="setup">
      <h2>Setup needed</h2>
      <ul>
        {failing.map((c) => (
          <li key={c.id}>{c.message}</li>
        ))}
      </ul>
    </section>
  );
}
