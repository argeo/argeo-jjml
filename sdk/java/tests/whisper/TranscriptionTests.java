package tests.whisper;

import java.io.IOException;
import java.lang.System.Logger.Level;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.argeo.jjml.sound.whisper.WhisperTranscription;
import org.argeo.jjml.whisper.WhisperCppContext;
import org.argeo.jjml.whisper.WhisperCppProcessor;

class TranscriptionTests extends AbstractWhisperTests {

	public TranscriptionTests(String modelHint) {
		super(modelHint);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		testJfkTranscription();
	}

	void testJfkTranscription() throws IOException {
		try (WhisperCppContext context = new WhisperCppContext(getModelPath())) {
			WhisperCppProcessor processor = new WhisperCppProcessor(context);
			String wavRelPath = "../native/tp/whisper.cpp/samples/jfk.wav";
			Path wavPath = Paths.get(wavRelPath);
			FloatBuffer floatBuf = WhisperTranscription.convert(wavPath);

			long begin = System.currentTimeMillis();
			String str = processor.transcribe(floatBuf);
			logger.log(Level.DEBUG, "Transcribed:\n" + str);
			logger.log(Level.DEBUG, "Transcription took " + (System.currentTimeMillis() - begin) + " ms");
		}
	}

}
