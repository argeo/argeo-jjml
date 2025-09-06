package org.argeo.jjml.whisper;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

public class WhisperCppProcessor {
	private final static float WHISPER_SAMPLE_RATE = 16000;

	private final WhisperCppContext context;

	public WhisperCppProcessor(WhisperCppContext context) {
		this.context = context;
	}

	private static native long doInit(byte[] modelPath, boolean useGpu, boolean flashAttention);

	private static native byte[] doFull(long contextPointer, FloatBuffer pcm, int offset, int length);

	public String transcribe(AudioInputStream input) throws IOException {
		doInit(null, false, false);
		FloatBuffer floatBuf = convert(input);
		byte[] utf8 = doFull(context.getAsLong(), floatBuf, 0, floatBuf.limit());
		return new String(utf8, StandardCharsets.UTF_8);
	}

	protected FloatBuffer convert(AudioInputStream input) throws IOException {
		AudioInputStream ais;
		if (input.getFormat().getFrameRate() == WHISPER_SAMPLE_RATE) {
			ais = input;
		} else {
			AudioFormat inputFormat = input.getFormat();
			System.out.println(inputFormat);
			AudioFormat outputFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED, WHISPER_SAMPLE_RATE,
					inputFormat.getSampleSizeInBits(), inputFormat.getChannels(), inputFormat.getFrameSize(),
					WHISPER_SAMPLE_RATE, true);
			System.out.println(outputFormat);
			boolean supported = AudioSystem.isConversionSupported(outputFormat, inputFormat);

			if (!supported)
				throw new IllegalArgumentException("Cannot convert audio");
			ais = AudioSystem.getAudioInputStream(outputFormat, input);
		}

		AudioFormat audioFormat = ais.getFormat();
		// TODO make it properly with buffer, etc
		ByteBuffer buf16i = ByteBuffer.wrap(ais.readAllBytes());
		buf16i.order(audioFormat.isBigEndian() ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);

		ByteBuffer buf32f = ByteBuffer.allocateDirect(buf16i.limit() * 2);
		// ByteBuffer buf32f = ByteBuffer.allocate(buf16i.limit() * 2);
		buf32f.order(audioFormat.isBigEndian() ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);

		ShortBuffer shortBuf = buf16i.asShortBuffer();
		FloatBuffer floatBuf = buf32f.asFloatBuffer();
		assert shortBuf.limit() == floatBuf.limit();
		for (int i = 0; i < shortBuf.limit(); i++) {
			float f = shortBuf.get();
			// TODO check the conversion very seriously

			float val;
			if (f < 0)
				val = -(f / Short.MIN_VALUE);
			else
				val = f / Short.MAX_VALUE;
			floatBuf.put(val);
			// System.out.println(val);
		}

		return floatBuf;
	}

	public static void main(String[] args) throws Exception {
		String modelId = "ggml-base.en.bin";
		//modelId = "ggml-small-q5_1.bin";
		Path modelPath = Paths.get(System.getProperty("user.home"),
				"dev/git/unstable/argeo-jjml/native/tp/whisper.cpp/models/", modelId);
		WhisperCppContext context = new WhisperCppContext(modelPath);
		WhisperCppProcessor processor = new WhisperCppProcessor(context);

		Path wavPath = Paths.get(System.getProperty("user.home"),
				"dev/git/unstable/argeo-jjml/native/tp/whisper.cpp/samples/jfk.wav");
		AudioInputStream inputStream = AudioSystem.getAudioInputStream(wavPath.toFile());
		String str = processor.transcribe(inputStream);
		System.out.println(str);
	}
}
