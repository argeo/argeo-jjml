package tests.llm;

import static java.lang.System.Logger.Level.DEBUG;

import java.io.IOException;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;

class LoadModelTests extends AbstractLlmTests {

	LoadModelTests(LlamaCppModel model) {
		super(model);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		testMetadata();
		testLoadUnloadDefaultContext();
	}

	void testMetadata() {
		LlamaCppModel model = getModel();
		logger.log(DEBUG, "Model " + model.getDescription());
		logger.log(DEBUG, model.getLayerCount() + " layers");
		logger.log(DEBUG, model.getEmbeddingSize() + " embedding size");
		logger.log(DEBUG, model.getVocabularySize() + " vocabulary size");
		logger.log(DEBUG, model.getContextTrainingSize() + " context training size");
		StringBuilder sb = new StringBuilder();
		for (String key : model.getMetadata().keySet())
			sb.append(key + "=" + model.getMetadata().get(key) + "\n");
		logger.log(DEBUG, "Metadata:\n" + sb);
	}

	void testLoadUnloadDefaultContext() {
		try (LlamaCppContext context = new LlamaCppContext(getModel());) {
			assert context.getContextSize() > 0;
		}
	}

}
