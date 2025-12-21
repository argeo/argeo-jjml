package org.argeo.jjml.whisper;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;

public class WhisperCppProcessor {
	public final static float WHISPER_SAMPLE_RATE = 16000;

	private final WhisperCppContext context;

	public WhisperCppProcessor(WhisperCppContext context) {
		this.context = context;
	}

	private static native byte[] doFull(long contextPointer, FloatBuffer pcm, int offset, int length);

	public String transcribe(FloatBuffer floatBuf) throws IOException {
		byte[] utf8 = doFull(context.getAsLong(), floatBuf, 0, floatBuf.limit());
		return new String(utf8, StandardCharsets.UTF_8);
	}

}
