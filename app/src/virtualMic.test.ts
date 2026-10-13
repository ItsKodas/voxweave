import { describe, expect, it } from "vitest";
import { heardAs, micOptionLabel } from "./virtualMic";

describe("virtual microphone names", () => {
  it("names the microphone other apps see for each known sending end", () => {
    expect(heardAs("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)")?.microphone).toBe("VoiceMeeter Output");
    expect(heardAs("VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)")?.microphone).toBe("VoiceMeeter Aux Output");
    expect(heardAs("VoiceMeeter VAIO3 Input (VB-Audio VoiceMeeter VAIO3)")?.microphone).toBe("VoiceMeeter VAIO3 Output");
    expect(heardAs("CABLE Input (VB-Audio Virtual Cable)")).toEqual({ microphone: "CABLE Output", note: null });
    expect(heardAs("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)")?.note).toMatch(/B1/);
    expect(heardAs("VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)")?.note).toMatch(/B2/);
  });

  it("leaves ordinary outputs alone", () => {
    expect(heardAs("Speakers (Logitech G733 Gaming Headset)")).toBeNull();
    expect(micOptionLabel("Speakers (Logitech G733 Gaming Headset)")).toBe("Speakers (Logitech G733 Gaming Headset)");
  });

  it("labels a choice by the microphone first", () => {
    expect(micOptionLabel("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)")).toBe("VoiceMeeter Output (sent through VoiceMeeter Input)");
    expect(micOptionLabel("CABLE Input (VB-Audio Virtual Cable)")).toBe("CABLE Output (sent through CABLE Input)");
  });
});
