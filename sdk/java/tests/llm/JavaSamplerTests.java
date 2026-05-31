package tests.llm;

import static java.lang.System.Logger.Level.DEBUG;
import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.LlamaCppSamplers.newJavaSampler;
import static org.argeo.jjml.llm.params.ContextParam.kv_unified;
import static org.argeo.jjml.llm.params.ContextParam.n_batch;
import static org.argeo.jjml.llm.params.ContextParam.n_ctx;

import java.io.IOException;
import java.util.Set;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppJavaSampler;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNativeSampler;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.LlamaCppTextProcessor;

class JavaSamplerTests extends AbstractLlmTests {
	JavaSamplerTests(LlamaCppModel model) {
		super(model);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		testGreedyJavaSampler();
	}

	void testGreedyJavaSampler() {
		LlamaCppModel model = getModel();
		Integer[] sequenceIds = { 1 };
		try ( //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 6144) //
						.with(n_batch, sequenceIds.length * 64) //
						.with(kv_unified, true) // required for robustness
				); //
				LlamaCppSamplerChain chain = new LlamaCppSamplerChain(
						newJavaSampler(new LlamaCppJavaSampler.SimpleGreedy())); //
				LlamaCppNativeSampler validatingSampler = LlamaCppSamplers.newSamplerGrammar(model, //
						"root ::= [ \\t\\n]* \"TEST\"", "root");//
		) {
//			long begin = System.currentTimeMillis();
			LlamaCppTextProcessor processor = new LlamaCppTextProcessor(context, chain, validatingSampler,
					Set.of(sequenceIds));

			String prompt = "Write HELLO\n"//
					+ "Hello\n"//
					+ "Write World\n"//
					+ "WORLD\n"//
					+ "Write test\n" //
			;
			logger.log(DEBUG, "=>\n" + prompt);
			String str = processor.processBatch(prompt);
			logger.log(DEBUG, "<=\n" + str);
			// System.out.println("\n\n## Processing took " + (System.currentTimeMillis() -
			// begin) + " ms");

		}
	}
}
