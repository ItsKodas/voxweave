"""What the installed app (AIVOICE_INSTALLED=1) says about the parts it does not have: training and text to
speech need the source checkout, and no message may send the user to a script that is not on their machine."""
import pytest

from aivoice import training, tts
from aivoice.devices import Device
from aivoice.setup_check import run_checks
from aivoice.training import TrainingError, TrainingManager
from aivoice.tts import TtsError, TtsManager
from tests.test_training import Rig as TrainingRig
from tests.test_tts import Rig as TtsRig

MIC = Device(1, "Headset Mic", 1, 0)


def honest(message: str) -> bool:
    lowered = message.lower()
    return ("not available in the installed app" in lowered and "source checkout" in lowered
            and "amd gpu" in lowered and "voice pack" in lowered
            and "bash" not in lowered and "setup.sh" not in lowered)


def test_both_messages_are_plain_and_name_no_script():
    assert honest(training.NOT_IN_INSTALLED_APP) and training.NOT_IN_INSTALLED_APP.startswith("Training is not")
    assert honest(tts.NOT_IN_INSTALLED_APP) and tts.NOT_IN_INSTALLED_APP.startswith("Text to speech is not")


@pytest.fixture
def training_rig(tmp_path):
    rig = TrainingRig(tmp_path)
    rig.manager = TrainingManager(
        rig.data, rig.library, rig.trainer, launch=rig.launch, verify=rig.verify,
        pid_alive=lambda pid: pid in rig.alive, now=lambda: rig.clock,
        is_worker=lambda pid, experiment: rig.workers.get(pid) == experiment, kill_pid=rig.killed.append,
        processes=rig.tools, sleep=rig.sleep, unavailable=training.NOT_IN_INSTALLED_APP)
    return rig


def test_the_training_environment_says_it_is_not_available_and_checks_nothing(training_rig):
    for refresh in (False, True):
        report = training_rig.manager.environment(refresh=refresh)
        assert report == {"ok": False, "ready": False, "unavailable": True, "checks": [
            {"id": "installed_app", "ok": False, "message": training.NOT_IN_INSTALLED_APP}]}
    assert training_rig.manager.environment(probe=False)["unavailable"] is True
    assert training_rig.verify_calls == []


def test_training_cannot_be_started_in_the_installed_app(training_rig):
    voice = training_rig.voice()
    with pytest.raises(TrainingError) as refused:
        training_rig.manager.start(voice)
    assert str(refused.value) == training.NOT_IN_INSTALLED_APP
    assert training_rig.verify_calls == [] and training_rig.launches == []
    assert training_rig.manager.status(voice)["state"] == "idle"
    assert not training_rig.manager.is_active()


@pytest.fixture
def tts_rig(tmp_path):
    rig = TtsRig(tmp_path)
    rig.manager = TtsManager(rig.data, rig.library, rig.tts_root, rig.player, lambda: rig.devices,
                             launch=rig.launch, verify=rig.verify, rvc_factory=rig.rvc, now=rig.now,
                             processes=rig.tools, tick=0.005, unavailable=tts.NOT_IN_INSTALLED_APP)
    yield rig
    rig.manager.shutdown()


def test_the_tts_environment_says_it_is_not_available_and_checks_nothing(tts_rig):
    for refresh in (False, True):
        report = tts_rig.manager.environment(refresh=refresh)
        assert report == {"ok": False, "ready": False, "unavailable": True, "checks": [
            {"id": "installed_app", "ok": False, "message": tts.NOT_IN_INSTALLED_APP}]}
    assert tts_rig.manager.environment(probe=False)["unavailable"] is True
    assert tts_rig.verify_calls == []


def test_speech_cannot_be_started_in_the_installed_app(tts_rig):
    with pytest.raises(TtsError) as refused:
        tts_rig.speak()
    assert str(refused.value) == tts.NOT_IN_INSTALLED_APP
    assert tts_rig.verify_calls == [] and tts_rig.launches == []
    assert not tts_rig.manager.is_active()


def test_the_source_checkout_is_unchanged(tmp_path):
    """Without the argument the managers check the environment as before."""
    rig = TrainingRig(tmp_path)
    rig.build()
    assert "unavailable" not in rig.manager.environment()
    assert rig.verify_calls == [False]


def _messages(tmp_path):
    return {c.id: c.message for c in run_checks(tmp_path, [MIC], [])}


def test_the_setup_check_of_the_installed_app_points_to_voice_packs_not_to_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("AIVOICE_INSTALLED", "1")
    messages = _messages(tmp_path)
    assert "voice pack" in messages["voice"] and ".voxweave" in messages["voice"]
    for word in ("tools/", "README", "Train one"):
        assert word not in messages["voice"]
    assert "start Voxweave again" in messages["base_model"] and "Download" not in messages["base_model"]
    assert "backend environment" not in messages["directml"] and "start Voxweave again" in messages["directml"]


def test_the_setup_check_of_the_source_checkout_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("AIVOICE_INSTALLED", raising=False)
    messages = _messages(tmp_path)
    assert "tools/export_rvc_onnx.py" in messages["voice"]
    assert "Download vec-768-layer-12.onnx into" in messages["base_model"]
    assert "Reinstall onnxruntime-directml in the backend environment" in messages["directml"]


def test_the_app_is_built_with_both_parts_unavailable_only_when_installed(monkeypatch):
    from aivoice import __main__ as entry

    monkeypatch.delenv("AIVOICE_INSTALLED", raising=False)
    assert entry.unavailable_parts() == (None, None)
    monkeypatch.setenv("AIVOICE_INSTALLED", "1")
    assert entry.unavailable_parts() == (training.NOT_IN_INSTALLED_APP, tts.NOT_IN_INSTALLED_APP)
