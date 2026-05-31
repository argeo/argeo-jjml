package tests.llm;

import static java.lang.System.Logger.Level.ERROR;
import static java.lang.System.Logger.Level.INFO;
import static java.lang.System.Logger.Level.WARNING;
import static org.argeo.jjml.llm.LlamaCppModel.defaultModelParams;

import java.lang.System.Logger;
import java.util.Arrays;
import java.util.List;
import java.util.function.BooleanSupplier;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.params.ModelParams;

import tests.AbstractJjmlTests;

public class AllLlmTests {
	private final static Logger logger = System.getLogger(AllLlmTests.class.getName());

	public void main(List<String> args) {
		try {
			if (!getClass().desiredAssertionStatus()) {
				logger.log(ERROR, "Assertions must be enabled. Please call Java with the -ea option.");
				return;
			}

			long begin = System.currentTimeMillis();

			// even without a model we can check whether native libraries are loading
			assert ((BooleanSupplier) () -> {
				LlamaCppNative.ensureLibrariesLoaded();
				return true;
			}).getAsBoolean();
			logger.log(INFO, "PASSED - Native libraries loaded");

			if (args.isEmpty()) {
				logger.log(WARNING, "No model was specified, only loading the native libraries was tested");
				return;
			}

			String arg0 = args.get(0);
			ModelParams modelParams = defaultModelParams();
			try (LlamaCppModel model = LoadModelTests.createModel(modelParams, arg0)) {

				new VocabularyTests(model).run();

				new LoadModelTests(model).run();
				new EmbeddingsTests(model).run();
				new BatchTests(model).run();
				new JavaSamplerTests(model).run();
				new InstructTests(model).run();
				new TravelAgentTests(model).run();
			}
			logger.log(INFO,
					"Tests passed in " + (System.currentTimeMillis() - begin) / 1000 + " s with model " + arg0);
		} catch (Exception | AssertionError e) {
			logger.log(ERROR, "Smoke tests failed", e);
			System.exit(2);
		} finally {
			LlamaCppBackend.destroy();
		}

		if (AbstractJjmlTests.allPassed)
			System.exit(0);
		else
			System.exit(1);
	}

	public static void main(String[] args) throws Exception {
		new AllLlmTests().main(Arrays.asList(args));
	}
}
