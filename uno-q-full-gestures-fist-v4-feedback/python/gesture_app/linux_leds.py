"""The UNO Q's two Linux-controlled RGB indicators (LED1 and LED2)."""

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)
CHANNELS = {
    "red": ("red:user", "red:panic"),
    "green": ("green:user", "green:wlan"),
    "blue": ("blue:user", "blue:bt"),
}


class LinuxRgbLeds:
    def __init__(self, root=Path("/sys/class/leds"), manage_triggers=False):
        self._root = Path(root)
        self._triggers = {}
        self._channels = [self._root / name for names in CHANNELS.values() for name in names]
        try:
            for channel in self._channels:
                if manage_triggers:
                    trigger = channel / "trigger"
                    active = next(word[1:-1] for word in trigger.read_text().split() if word.startswith("["))
                    self._triggers[trigger] = active
                    trigger.write_text("none\n")
                (channel / "brightness").write_text("0\n")
        except (OSError, StopIteration) as exc:
            self.close()
            raise RuntimeError("Cannot control UNO Q RGB LEDs in /sys/class/leds. Run on the board with write access to LED brightness/trigger files; disable led.enabled for a desktop test.") from exc

    def set_color(self, color):
        if color not in ("off", "red", "green", "blue"):
            raise ValueError(f"Unsupported LED color: {color}")
        # Clear other channels first to avoid momentarily mixing colors.
        for channel_color, names in CHANNELS.items():
            if channel_color != color:
                for name in names:
                    (self._root / name / "brightness").write_text("0\n")
        if color != "off":
            for name in CHANNELS[color]:
                (self._root / name / "brightness").write_text("1\n")

    def close(self):
        for channel in self._channels:
            try:
                (channel / "brightness").write_text("0\n")
            except OSError as exc:
                LOG.warning("Could not clear %s: %s", channel.name, exc)
        # Restore Wi-Fi/Bluetooth indication after standalone recognition exits.
        for trigger, original in self._triggers.items():
            try:
                trigger.write_text(original + "\n")
            except OSError as exc:
                LOG.warning("Could not restore %s: %s", trigger, exc)
        self._triggers.clear()


def connect_linux_leds(config):
    if not config.data["led"]["enabled"]:
        return None
    # App Lab manages system triggers outside its container; standalone owns them.
    return LinuxRgbLeds(manage_triggers=config.runtime == "standalone")
