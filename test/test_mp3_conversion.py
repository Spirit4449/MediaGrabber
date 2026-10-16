import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import daily_bns_sync as bns


class MP3ConversionTests(unittest.TestCase):
    def test_boundary_uses_supported_bitrate_with_headroom(self):
        duration = bns.BNS_MAX_UPLOAD_BYTES * 8 / 129000
        with patch.object(bns, 'get_audio_duration_seconds', return_value=duration):
            self.assertEqual(bns.mp3_bitrate_for_upload(Path('audio.m4a')), 112)

    def test_long_recording_can_use_less_than_64_kbps(self):
        with patch.object(bns, 'get_audio_duration_seconds', return_value=14400):
            self.assertEqual(bns.mp3_bitrate_for_upload(Path('audio.m4a')), 24)

    def test_oversize_retries_using_measured_size(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'audio.m4a'
            source.touch()
            output = source.with_suffix('.mp3')
            calls = []
            def encode(cmd, **kwargs):
                calls.append(cmd)
                output.write_bytes(b'x' * (1001 if len(calls) == 1 else 900))
                return SimpleNamespace(returncode=0)
            with patch.object(bns, 'BNS_MAX_UPLOAD_BYTES', 1000), \
                 patch.object(bns.shutil, 'which', return_value='/ffmpeg'), \
                 patch.object(bns, 'mp3_bitrate_for_upload', return_value=128), \
                 patch.object(bns.subprocess, 'run', side_effect=encode):
                self.assertEqual(bns.convert_audio_to_mp3(source), output)
            self.assertIn('128k', calls[0])
            self.assertIn('112k', calls[1])

    def test_missing_duration_falls_back_after_measuring_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'audio.m4a'
            source.touch()
            calls = []
            def encode(cmd, **kwargs):
                calls.append(cmd)
                source.with_suffix('.mp3').write_bytes(b'x' * (2000 if len(calls) == 1 else 900))
                return SimpleNamespace(returncode=0)
            with patch.object(bns, 'BNS_MAX_UPLOAD_BYTES', 1000), \
                 patch.object(bns.shutil, 'which', return_value='/ffmpeg'), \
                 patch.object(bns, 'get_audio_duration_seconds', return_value=None), \
                 patch.object(bns.subprocess, 'run', side_effect=encode):
                bns.convert_audio_to_mp3(source)
            self.assertIn('-q:a', calls[0])
            self.assertIn('80k', calls[1])
