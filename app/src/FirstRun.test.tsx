import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FirstRun, useSetup } from "./FirstRun";
import { SetupState } from "./setupApi";

vi.mock("./setupApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./setupApi")>()),
  setupApi: { state: vi.fn(), start: vi.fn(), onProgress: vi.fn() },
}));
import { installedApp, rememberMode, setupApi } from "./setupApi";

const state = (over: Partial<SetupState> = {}): SetupState => ({
  mode: "installed", status: "running", step: "packages", percent: 37,
  message: "Downloading and installing the voice engine's packages", error: null,
  root: "C:\\Users\\me\\AppData\\Local\\Voxweave", data_dir: "C:\\Users\\me\\AppData\\Local\\Voxweave\\data",
  setup_log: "C:\\Users\\me\\AppData\\Local\\Voxweave\\runtime\\setup.log", ...over,
});
const DEV = state({ mode: "dev", status: "ready", step: "", percent: 100, message: "", root: "", data_dir: "", setup_log: "" });

describe("FirstRun", () => {
  it("says what is downloaded, how much, that it happens once and needs the internet", () => {
    render(<FirstRun setup={state()} onRetry={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "First-time setup" })).toBeInTheDocument();
    const about = screen.getByText(/is downloading what its voice engine needs/);
    expect(about).toHaveTextContent("Python");
    expect(about).toHaveTextContent("packages");
    expect(about).toHaveTextContent("base speech model");
    expect(about).toHaveTextContent("about 525 MB");
    expect(screen.getByText(/needs an internet connection this once/i)).toBeInTheDocument();
    expect(screen.getByText(/C:\\Users\\me\\AppData\\Local\\Voxweave/)).toBeInTheDocument();
  });

  it("shows the current step and how far the setup is", () => {
    render(<FirstRun setup={state()} onRetry={vi.fn()} />);
    const bar = screen.getByRole("progressbar", { name: "Setup progress" });
    expect(bar).toHaveAttribute("value", "37");
    expect(bar).toHaveAttribute("max", "100");
    expect(screen.getByText("Downloading and installing the voice engine's packages")).toBeInTheDocument();
    expect(screen.getByText("37%")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it("shows the error, where the details are, and Retry after a failure", async () => {
    const onRetry = vi.fn();
    render(<FirstRun setup={state({ status: "failed", error: "Could not download the base speech model: timed out." })} onRetry={onRetry} />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Could not download the base speech model: timed out.");
    expect(alert).toHaveTextContent("runtime\\setup.log");
    expect(screen.queryByText("Downloading and installing the voice engine's packages")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("keeps the bar where the setup stopped, since a retry continues from there", () => {
    render(<FirstRun setup={state({ status: "failed", error: "x", percent: 70 })} onRetry={vi.fn()} />);
    expect(screen.getByRole("progressbar", { name: "Setup progress" })).toHaveAttribute("value", "70");
  });
});

function Probe() {
  const { setup, retry } = useSetup();
  return (
    <>
      <p>{setup === null ? "unknown" : `${setup.mode} ${setup.status} ${setup.percent}`}</p>
      <button onClick={retry}>again</button>
    </>
  );
}

describe("useSetup", () => {
  let push: (s: SetupState) => void;
  let unlisten: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.resetAllMocks();
    rememberMode(null);
    unlisten = vi.fn();
    vi.mocked(setupApi.onProgress).mockImplementation(async (handler) => {
      push = handler;
      return unlisten as unknown as () => void;
    });
    vi.mocked(setupApi.state).mockResolvedValue(state());
    vi.mocked(setupApi.start).mockResolvedValue(state({ percent: 0, step: "python" }));
  });

  it("asks the shell for the state and then follows its events", async () => {
    render(<Probe />);
    expect(screen.getByText("unknown")).toBeInTheDocument();
    expect(await screen.findByText("installed running 37")).toBeInTheDocument();
    act(() => push(state({ percent: 80 })));
    expect(screen.getByText("installed running 80")).toBeInTheDocument();
    act(() => push(state({ status: "ready", percent: 100 })));
    expect(screen.getByText("installed ready 100")).toBeInTheDocument();
  });

  it("does not let a slow first answer overwrite a newer event", async () => {
    let answer: (s: SetupState) => void = () => {};
    vi.mocked(setupApi.state).mockReturnValue(new Promise((resolve) => { answer = resolve; }));
    render(<Probe />);
    await act(async () => {});
    act(() => push(state({ percent: 80 })));
    await act(async () => answer(state({ percent: 10 })));
    expect(screen.getByText("installed running 80")).toBeInTheDocument();
  });

  it("retry starts the setup again and shows the state the shell returns", async () => {
    vi.mocked(setupApi.state).mockResolvedValue(state({ status: "failed", error: "x" }));
    render(<Probe />);
    await screen.findByText("installed failed 37");
    await userEvent.click(screen.getByRole("button", { name: "again" }));
    expect(setupApi.start).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("installed running 0")).toBeInTheDocument();
  });

  it("stays unknown when there is no shell to ask, and remembers nothing", async () => {
    vi.mocked(setupApi.state).mockRejectedValue(new Error("no shell"));
    vi.mocked(setupApi.onProgress).mockRejectedValue(new Error("no shell"));
    render(<Probe />);
    await act(async () => {});
    expect(screen.getByText("unknown")).toBeInTheDocument();
    expect(installedApp()).toBe(false);
  });

  it("remembers for the panels whether this is the installed app", async () => {
    render(<Probe />);
    await screen.findByText("installed running 37");
    expect(installedApp()).toBe(true);
    act(() => push(DEV));
    expect(installedApp()).toBe(false);
  });

  it("stops listening when it goes away", async () => {
    const view = render(<Probe />);
    await screen.findByText("installed running 37");
    view.unmount();
    await act(async () => {});
    expect(unlisten).toHaveBeenCalledTimes(1);
  });
});
