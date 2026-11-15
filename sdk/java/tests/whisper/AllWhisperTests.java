package tests.whisper;

import static java.lang.System.Logger.Level.ERROR;
import static java.lang.System.Logger.Level.INFO;

import java.lang.System.Logger;
import java.util.Arrays;
import java.util.List;

import org.argeo.jjml.whisper.WhisperNative;

import tests.AbstractJjmlTests;

public class AllWhisperTests {
	private final static String DEFAULT_MODEL_HINT = "ggml-medium-q5_0.bin";
	private final static Logger logger = System.getLogger(AllWhisperTests.class.getName());

	public void main(List<String> args) {
		try {
			if (!getClass().desiredAssertionStatus()) {
				throw new IllegalStateException("Assertions must be enabled. Please call Java with the -ea option.");
			}

			long begin = System.currentTimeMillis();

			// even without a model we can check whether native libraries are loading
			WhisperNative.ensureLibrariesLoaded();
			logger.log(INFO, "PASSED - Native libraries loaded");

			String modelId = DEFAULT_MODEL_HINT;
//			modelId = "ggml-base.en.bin";
//			modelId = "ggml-base.bin";
//			modelId = "ggml-base-q8_0.bin";
//			modelId = "ggml-small-q5_1.bin";
			modelId = "ggml-medium-q5_0.bin";
//			modelId = "ggml-medium-q8_0.bin";
//			modelId = "ggml-large-v3.bin";
//			modelId = "ggml-large-v3-q5_0.bin";

			String modelHint = args.isEmpty() ? modelId : args.get(0);

			new TranscriptionTests(modelHint).run();

			logger.log(INFO,
					"Tests passed in " + (System.currentTimeMillis() - begin) / 1000 + " s with model " + modelHint);
		} catch (Exception e) {
			logger.log(ERROR, "Tests could not run", e);
			e.printStackTrace();
			System.exit(2);
		} finally {
		}

		if (AbstractJjmlTests.allPassed())
			System.exit(0);
		else
			System.exit(1);
	}

	public static void main(String[] args) throws Exception {
		new AllWhisperTests().main(Arrays.asList(args));
	}

}
