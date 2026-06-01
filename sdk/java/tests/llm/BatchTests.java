package tests.llm;

import static java.lang.System.Logger.Level.DEBUG;
import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.params.ContextParam.kv_unified;
import static org.argeo.jjml.llm.params.ContextParam.n_batch;
import static org.argeo.jjml.llm.params.ContextParam.n_ctx;

import java.io.IOException;
import java.util.Set;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNativeSampler;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.LlamaCppTextProcessor;

class BatchTests extends AbstractLlmTests {
	BatchTests(LlamaCppModel model) {
		super(model);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		testGrammarBatch();
	}

	void testGrammarBatch() {
		LlamaCppModel model = getModel();

		String prompt = "Write HELLO\n"//
				+ "HELLO\n"//
				+ "Write WORLD\n"//
				+ "WORLD\n"//
				+ "Write TEST\n" //
		;

		// !! max seq_id must be < 64
		// TODO understand why
		Integer[] sequenceIds = { 1, 10, 63 };
		try ( //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 6144) //
						.with(n_batch, sequenceIds.length * prompt.length()) //
						.with(kv_unified, true) // required for robustness
				); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
				LlamaCppNativeSampler validatingSampler = LlamaCppSamplers.newSamplerGrammar(model, //
						"root ::= [ \\t\\n]* \"TEST\"", "root");//
		) {
//			long begin = System.currentTimeMillis();
			LlamaCppTextProcessor processor = new LlamaCppTextProcessor(context, chain, validatingSampler,
					Set.of(sequenceIds));

			logger.log(DEBUG, "=>\n" + prompt);
			String str = processor.processBatch(prompt);
			logger.log(DEBUG, "<=\n" + str);
			// System.out.println("\n\n## Processing took " + (System.currentTimeMillis() -
			// begin) + " ms");

		}
	}

	public static void main(String[] args) throws Exception {
		String hint = args.length == 0 ? "allenai/OLMo-2-0425-1B-Instruct-GGUF" : args[0];
		new BatchTests(AbstractLlmTests.createModel(hint)).all();
	}
}
