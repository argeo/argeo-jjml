package examples;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.TargetDataLine;

import org.argeo.jjml.sound.whisper.WhisperTranscription;
import org.argeo.jjml.whisper.WhisperCppContext;
import org.argeo.jjml.whisper.WhisperCppProcessor;
import org.argeo.jjml.whisper.WhisperNative;

public class JjmlWhisperRecorder {
	private final static String DEFAULT_MODEL_HINT = "ggml-medium-q5_0.bin";

	private static final int totalSize = 16000 * 10;
	private static int totalWritten = 0;

	public static void main(String[] args) throws Exception {
		WhisperNative.ensureLibrariesLoaded(); // fail fast

		Path tempWav = Files.createTempFile(JjmlWhisperRecorder.class.getName(), ".wav");

		try {
			ByteArrayOutputStream out = new ByteArrayOutputStream(totalSize);

			AudioFormat format = new AudioFormat(16000.0f, 16, 1, true, true);
			TargetDataLine microphone = AudioSystem.getTargetDataLine(format);

			int frameSizeInBytes = format.getFrameSize();
			int bufferLengthInFrames = microphone.getBufferSize() / 8;
			final int bufferLengthInBytes = bufferLengthInFrames * frameSizeInBytes;

			microphone.open(format, microphone.getBufferSize());

			System.out.println("Start recording... (~4s)");
			microphone.start();
			Thread.sleep(1000);
			long begin = System.currentTimeMillis();
			buildByteOutputStream(out, microphone, bufferLengthInBytes);
			microphone.stop();
			System.out.println("Stopped recording after " + (System.currentTimeMillis() - begin) / 1000 + " s \n\n");

			byte[] buf = out.toByteArray();
			AudioInputStream ais = new AudioInputStream(new ByteArrayInputStream(buf), format,
					buf.length / format.getFrameSize());

			AudioSystem.write(ais, AudioFileFormat.Type.WAVE, tempWav.toFile());
			ais.close();
			System.out.println("Wrote " + tempWav);

			String modelId = DEFAULT_MODEL_HINT;
			// modelId = "ggml-base.en.bin";
			// modelId = "ggml-base.bin";
			// modelId = "ggml-base-q8_0.bin";
			// modelId = "ggml-small-q5_1.bin";
			// modelId = "ggml-medium-q5_0.bin";
			// modelId = "ggml-medium-q8_0.bin";
			// modelId = "ggml-large-v3.bin";
			// modelId = "ggml-large-v3-q5_0.bin";

			Path modelPath = Paths.get(System.getProperty("user.home"), "dev/foss/AI/openai/converted", modelId);
			try (WhisperCppContext context = new WhisperCppContext(modelPath)) {
				WhisperCppProcessor processor = new WhisperCppProcessor(context);
				FloatBuffer floatBuf = WhisperTranscription.convert(tempWav);
				String transcribed = processor.transcribe(floatBuf);

				System.out.println("----------------------------------------------");
				System.out.println(transcribed);
				System.out.println("----------------------------------------------");
			}
		} finally {
			try {
				Thread.sleep(1000);// wait before deleting (Windows)
				Files.delete(tempWav);
				System.out.println("Deleted " + tempWav);
			} catch (IOException e) {
				System.err.println(e.getMessage());
				tempWav.toFile().deleteOnExit();
			}
		}
	}

	private static void buildByteOutputStream(final ByteArrayOutputStream out, final TargetDataLine line,
			final int bufferLengthInBytes) throws IOException {
		final byte[] data = new byte[bufferLengthInBytes];
		int numBytesRead;
		while (true) {
			if ((numBytesRead = line.read(data, 0, bufferLengthInBytes)) == -1) {
				break;
			}
			out.write(data, 0, numBytesRead);
			totalWritten = totalWritten + numBytesRead;
			if (totalWritten > totalSize)
				break;
		}
	}

}
