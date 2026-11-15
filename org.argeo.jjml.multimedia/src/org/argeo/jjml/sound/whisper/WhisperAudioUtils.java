package org.argeo.jjml.sound.whisper;

import javax.sound.sampled.AudioFormat;

public class WhisperAudioUtils {
	AudioFormat whisperNativeFormat = new AudioFormat(//
			AudioFormat.Encoding.PCM_FLOAT, // Encoding: 32-bit Floating Point
			16000.0f, // Sample Rate: 16kHz
			32, // Sample Size: 32 bits
			1, // Channels: 1 (Mono)
			4, // Frame Size: 4 bytes
			16000.0f, // Frame Rate
			false // Big Endian: false
	);

	AudioFormat whisperWavFormat = new AudioFormat(//
			AudioFormat.Encoding.PCM_SIGNED, // Encoding
			16000.0f, // Sample Rate: strictly 16kHz
			16, // Sample Size: 16 bits
			1, // Channels: 1 (Mono)
			2, // Frame Size: 2 bytes (1 chan * 2 bytes)
			16000.0f, // Frame Rate: 16k frames/sec
			false // Big Endian: false (Little Endian)
	);

	/** singleton */
	private WhisperAudioUtils() {
	}
}
