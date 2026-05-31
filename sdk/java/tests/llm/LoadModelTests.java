package tests.llm;

import static java.lang.System.Logger.Level.DEBUG;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.util.SimpleProgressCallback;

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

	static LlamaCppModel createModel(String hint) throws IOException {
		return createModel(LlamaCppModel.defaultModelParams(), hint);
	}

	static LlamaCppModel createModel(ModelParams modelParams, String hint) throws IOException {
		Path modelPath = Paths.get(hint);
		if (!Files.exists(modelPath))
			modelPath = new HfModelCache().getLocalFile(hint);
		if (!Files.exists(modelPath))
			throw new IllegalArgumentException("Could not find GGUF model " + modelPath);
		return loadModel(modelParams, modelPath);
	}

	static LlamaCppModel loadModel(ModelParams modelParams, Path modelPath) throws IOException {
		SimpleProgressCallback progressCallback = System.getLogger(LoadModelTests.class.getName()).isLoggable(DEBUG)
				? new SimpleProgressCallback()
				: null;
		Future<LlamaCppModel> loaded = LlamaCppModel.loadAsync(modelPath, modelParams, progressCallback, null);
		try {
			LlamaCppModel model = loaded.get();
			return model;
		} catch (InterruptedException | ExecutionException e) {
			throw new IllegalStateException("Could not load model " + modelPath, e);
		}

	}

}
