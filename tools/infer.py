import torch
import moduleconf
import numpy as np
from io import BytesIO
from transkun.Data import writeMidi
from transkun.Util import computeParamSize


class TranskunInfer:
    def __init__(self, weight_path=None, conf_path=None, device="cpu"):
        import pkg_resources

        # Define default paths
        self.default_weight = pkg_resources.resource_filename(__name__, "../transkun/pretrained/2.0.pt")
        self.default_conf = pkg_resources.resource_filename(__name__, "../transkun/pretrained/2.0.conf")

        # Use provided paths or defaults
        self.weight_path = weight_path or self.default_weight
        self.conf_path = conf_path or self.default_conf
        self.device = device

        # Load configuration and model
        self._load_model()

    def _load_model(self):
        # Load configuration
        self.conf_manager = moduleconf.parseFromFile(self.conf_path)
        TransKun = self.conf_manager["Model"].module.TransKun
        conf = self.conf_manager["Model"].config

        # Load model checkpoint
        checkpoint = torch.load(self.weight_path, map_location=self.device)
        self.model = TransKun(conf=conf).to(self.device)

        if "best_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["best_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.eval()

    def get_midi(self, audio: np.ndarray, fs: int = 44100, segment_hop_size: float = None, segment_size: float = None) -> bytes:
        """
        Convert audio (numpy array) to MIDI bytes.

        Args:
            audio (np.ndarray): Input audio signal as a NumPy array.
            fs (int): Sampling rate of the audio signal.
            segment_hop_size (float): Hop size in seconds for processing. If None, uses model default.
            segment_size (float): Segment size in seconds for processing. If None, uses model default.

        Returns:
            bytes: MIDI file content as bytes.
        """
        import soxr

        # Resample if necessary
        if fs != self.model.fs:
            audio = soxr.resample(audio, fs, self.model.fs)

        # Convert to PyTorch tensor
        x = torch.from_numpy(audio).to(self.device)

        # Perform transcription
        torch.set_grad_enabled(False)
        notes_est = self.model.transcribe(
            x, stepInSecond=segment_hop_size, segmentSizeInSecond=segment_size, discardSecondHalf=False
        )

        # Write MIDI to memory
        output_midi = writeMidi(notes_est)
        midi_buffer = BytesIO()
        output_midi.write(midi_buffer)
        midi_buffer.seek(0)

        return midi_buffer.read()
