package tests.llm;

import static java.lang.System.Logger.Level.ERROR;
import static java.lang.System.Logger.Level.INFO;
import static org.argeo.jjml.llm.LlamaCppModel.defaultModelParams;

import java.lang.System.Logger;
import java.util.Arrays;
import java.util.List;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.params.ModelParams;

import tests.AbstractJjmlTests;

public class AllLlmTests {
	private final static Logger logger = System.getLogger(AllLlmTests.class.getName());

	private final static String DEFAULT_MODEL_HINT = "mistralai/Ministral-3-3B-Instruct-2512-GGUF";

	public void main(List<String> args) {
		try {
			if (!getClass().desiredAssertionStatus()) {
				throw new IllegalStateException("Assertions must be enabled. Please call Java with the -ea option.");
			}

			long begin = System.currentTimeMillis();

			// even without a model we can check whether native libraries are loading
			LlamaCppNative.ensureLibrariesLoaded();
			logger.log(INFO, "PASSED - Native libraries loaded");

			String modelHint = args.isEmpty() ? DEFAULT_MODEL_HINT : args.get(0);
			ModelParams modelParams = defaultModelParams();
			try (LlamaCppModel model = AbstractLlmTests.createModel(modelParams, modelHint)) {

				new VocabularyTests(model).run();

				new LoadModelTests(model).run();
				new EmbeddingsTests(model).run();
				new BatchTests(model).run();
				new JavaSamplerTests(model).run();
				new InstructTests(model).run();
				new TravelAgentTests(model).run();
			}
			logger.log(INFO,
					"Tests passed in " + (System.currentTimeMillis() - begin) / 1000 + " s with model " + modelHint);
		} catch (Exception e) {
			logger.log(ERROR, "Tests could not run", e);
			e.printStackTrace();
			System.exit(2);
		} finally {
			LlamaCppBackend.destroy();
		}

		if (AbstractJjmlTests.allPassed())
			System.exit(0);
		else
			System.exit(1);
	}

	public static void main(String[] args) throws Exception {
		new AllLlmTests().main(Arrays.asList(args));
	}
}
