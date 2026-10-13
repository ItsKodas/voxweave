/**
 * A virtual microphone has two ends with different names. An app can only play into the sending end, which
 * Windows lists under playback ("VoiceMeeter Input", "CABLE Input"). Other apps pick the receiving end under
 * microphones ("VoiceMeeter Output", "CABLE Output"). People think of the microphone, so the screen names that.
 */
export type HeardAs = { microphone: string; note: string | null };

const VOICEMEETER_NOTE = (bus: string) =>
  `VoiceMeeter must be running, with ${bus} switched on for the strip this app plays into.`;

/** The microphone other apps see for a playback device, or null when the device is not a known virtual mic. */
export function heardAs(playbackName: string): HeardAs | null {
  const name = playbackName.toLowerCase();
  if (name.includes("cable input")) return { microphone: "CABLE Output", note: null };
  if (name.includes("voicemeeter aux input")) return { microphone: "VoiceMeeter Aux Output", note: VOICEMEETER_NOTE("B2") };
  if (name.includes("voicemeeter vaio3 input")) return { microphone: "VoiceMeeter VAIO3 Output", note: VOICEMEETER_NOTE("B3") };
  if (name.includes("voicemeeter input")) return { microphone: "VoiceMeeter Output", note: VOICEMEETER_NOTE("B1") };
  return null;
}

/** What a virtual mic choice is called in a list: the microphone other apps see, then the device Windows plays into. */
export function micOptionLabel(playbackName: string): string {
  const heard = heardAs(playbackName);
  return heard ? `${heard.microphone} (sent through ${playbackName.split(" (")[0]})` : playbackName;
}
