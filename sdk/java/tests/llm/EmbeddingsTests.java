package tests.llm;

import static java.lang.System.Logger.Level.DEBUG;
import static org.argeo.jjml.llm.params.ContextParam.embeddings;
import static org.argeo.jjml.llm.params.ContextParam.kv_unified;
import static org.argeo.jjml.llm.params.ContextParam.n_batch;
import static org.argeo.jjml.llm.params.ContextParam.n_ctx;
import static org.argeo.jjml.llm.params.ContextParam.n_ubatch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppEmbeddingProcessor;
import org.argeo.jjml.llm.LlamaCppModel;

class EmbeddingsTests extends AbstractLlmTests {
	EmbeddingsTests(LlamaCppModel model) {
		super(model);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		testSimpleEmbeddings();
	}

	void testSimpleEmbeddings() {
		int batchSize = 512;
		try (LlamaCppContext context = new LlamaCppContext(getModel(), LlamaCppContext.defaultContextParams() //
				.with(embeddings, true) //
				.with(n_ctx, 6144) //
				.with(n_batch, batchSize) //
				.with(n_ubatch, batchSize) // must be same for embeddings
				.with(kv_unified, true) // required for robustness
		);) {
			LlamaCppEmbeddingProcessor embeddingProcessor = new LlamaCppEmbeddingProcessor(context);

			List<String> prompts = new ArrayList<>();
			prompts.add("Hello world!");
			prompts.add("Good night and good luck.");
			for (String s : prompts)
				logger.log(DEBUG, "=>\n" + s);

			float[][] embeddings = embeddingProcessor.processEmbeddings(prompts);
			assert embeddings.length != 0;

			for (float[] embedding : embeddings) {
				logger.log(DEBUG, "<=\n[ " + embedding[0] + ", " + embedding[1] + ", ... ]");
			}
		}
	}
}
