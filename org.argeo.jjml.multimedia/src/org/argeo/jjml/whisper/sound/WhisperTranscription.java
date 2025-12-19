package org.argeo.jjml.whisper.sound;

import static org.argeo.jjml.whisper.WhisperCppProcessor.WHISPER_SAMPLE_RATE;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

import org.argeo.jjml.whisper.WhisperCppContext;
import org.argeo.jjml.whisper.WhisperCppProcessor;

public class WhisperTranscription {

	public static FloatBuffer convert(AudioInputStream input) throws IOException {
		AudioInputStream ais;
		if (input.getFormat().getFrameRate() == WHISPER_SAMPLE_RATE) {
			ais = input;
		} else {
			AudioFormat inputFormat = input.getFormat();
			System.out.println(inputFormat);
			AudioFormat outputFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED, WHISPER_SAMPLE_RATE,
					inputFormat.getSampleSizeInBits(), inputFormat.getChannels(), inputFormat.getFrameSize(),
					WHISPER_SAMPLE_RATE, inputFormat.isBigEndian());
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
		modelId = "ggml-base.bin";
//		modelId = "ggml-base-q8_0.bin";
//		modelId = "ggml-small-q5_1.bin";
//		modelId = "ggml-medium-q5_0.bin";
		modelId = "ggml-medium-q8_0.bin";
//		modelId = "ggml-large-v3.bin";
//		modelId = "ggml-large-v3-q5_0.bin";

		Path modelPath = Paths.get("/srv/ml/models", modelId);
		WhisperCppContext context = new WhisperCppContext(modelPath);
		WhisperCppProcessor processor = new WhisperCppProcessor(context);

		String wavRelPath = "dev/git/unstable/argeo-jjml/native/tp/whisper.cpp/samples/jfk.wav";
		wavRelPath = "Music/18juin/cdg.wav";
//		wavRelPath = "Music/18juin/cdg-remastered.wav";
//		wavRelPath = "Music/18juin/cdg-48kHz.wav";
		Path wavPath = Paths.get(System.getProperty("user.home"), wavRelPath);
		AudioInputStream inputStream = AudioSystem.getAudioInputStream(wavPath.toFile());
		FloatBuffer floatBuf = convert(inputStream);

		long begin = System.currentTimeMillis();
		String str = processor.transcribe(floatBuf);
		System.out.println(str);
		System.err.println("Transcription took " + (System.currentTimeMillis() - begin) + " ms");
	}

}
